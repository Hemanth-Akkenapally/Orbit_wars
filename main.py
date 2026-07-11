"""BioHub_Project baseline agent.

The strategy is intentionally explicit:
1. grow production with exact neutral captures,
2. chain expansion from captured hubs,
3. defend valuable hubs,
4. attack only when the economy or local math is favorable.
"""

from __future__ import annotations

import math


CENTER_X = 50.0
CENTER_Y = 50.0
BOARD_CENTER = (CENTER_X, CENTER_Y)
SUN_RADIUS = 10.0
MIN_SEND = 3
LOOKAHEAD = 80
MAX_ACTIONS = 24
EPS = 1e-9

MEMORY = {
    "turn_by_player": {},
    "pending_by_player": {},
}


def _agent_impl(obs, config=None):
    try:
        state = parse_state(obs, config)
        if not state["planets"] or state["turn"] <= 0:
            update_memory(state)
            return []
        update_memory(state)
        actions, plans = build_plan(state)
        cleaned, cleaned_plans = sanitize_actions(actions, plans, state)
        remember_pending(cleaned_plans, state)
        return cleaned
    except Exception:
        return fallback_action(obs, config)


def build_plan(state):
    mine = my_planets(state)
    if not mine:
        return [], []

    budgets = {p["id"]: available_ships(p, state) for p in mine}
    sources = sorted(
        mine,
        key=lambda p: (budgets[p["id"]], p["production"], p["ships"]),
        reverse=True,
    )
    action_limit = action_limit_for_turn(state)
    actions = []
    plans = []

    group = group_capture_plan(sources, budgets, state, action_limit)
    if group:
        return group["actions"], group["plans"]

    for source in sources:
        sid = source["id"]
        while budgets.get(sid, 0) >= MIN_SEND and len(actions) < action_limit:
            mission = choose_mission(source, budgets[sid], state)
            if mission is None:
                break
            target, ships, reason = mission
            max_launch = max(0, source["ships"] - 1)
            ships = int(max(MIN_SEND, min(ships, budgets[sid], max_launch)))
            if ships < MIN_SEND:
                break
            solution = shot_solution(source, target, ships, state)
            if solution is None:
                mark_bad_target(target, ships)
                break
            angle, eta = solution
            action = [source["id"], angle, ships]
            actions.append(action)
            plans.append(make_plan(source, target, ships, eta, reason))
            budgets[sid] -= ships
            reserve_target(target, ships, reason)
            if reason in ("expand", "attack", "steal"):
                break
            if reason in ("rescue", "reinforce"):
                break
    return actions, plans


def choose_mission(source, budget, state):
    rescue = best_rescue(source, budget, state)
    if rescue:
        return rescue

    neutral = best_neutral_capture(source, budget, state)
    if neutral:
        return neutral

    attack = best_enemy_attack(source, budget, state)
    if attack:
        return attack

    reinforce = best_reinforce(source, budget, state)
    if reinforce:
        return reinforce
    return None


# ---------------------------------------------------------------------------
# Parsing and state memory


def parse_state(obs, config):
    me_raw = first_present(obs, ("player", "mark", "player_id", "id"), None)
    me = int(me_raw) if me_raw is not None else 0
    angular_velocity = float(first_present(obs, ("angular_velocity",), 0.0))
    raw_planets = as_list(first_present(obs, ("planets", "map", "entities"), []))
    raw_fleets = as_list(first_present(obs, ("fleets", "ships", "moving_fleets"), []))
    comet_ids = set(as_list(first_present(obs, ("comet_planet_ids",), [])))

    planets = [normalise_planet(p, i, angular_velocity, comet_ids) for i, p in enumerate(raw_planets)]
    fleets = [normalise_fleet(f, i) for i, f in enumerate(raw_fleets)]
    infer_fleet_targets(planets, fleets)

    if me_raw is None and planets:
        owners = sorted({p["owner"] for p in planets if p["owner"] not in (-1, None)})
        if owners:
            me = owners[0]

    turn_raw = first_present(obs, ("step", "turn"), None)
    previous = MEMORY.setdefault("turn_by_player", {}).get(me, -1)
    if turn_raw is None:
        turn = 0 if looks_initial(planets, fleets, me) else previous + 1
    else:
        turn = int(turn_raw)
    if turn <= 0 or turn < previous:
        MEMORY.setdefault("pending_by_player", {}).pop(me, None)

    players = sorted({p["owner"] for p in planets if p["owner"] not in (-1, None)})
    return {
        "me": me,
        "turn": turn,
        "planets": planets,
        "fleets": fleets,
        "players": players,
        "crowded": len(players) >= 4,
        "episode_steps": int(first_present(config, ("episodeSteps", "episode_steps"), 500)),
        "ship_speed": float(first_present(config, ("shipSpeed", "ship_speed"), 6.0)),
        "sun_radius": float(first_present(config, ("sunRadius", "sun_radius"), SUN_RADIUS)),
    }


def normalise_planet(raw, index, angular_velocity, comet_ids):
    if isinstance(raw, (list, tuple)) and len(raw) >= 7:
        pid, owner, x, y, radius, ships, production = raw[:7]
    elif isinstance(raw, dict):
        pid = raw.get("id", index)
        owner = raw.get("owner", raw.get("player", -1))
        x = raw.get("x", raw.get("cx", 0.0))
        y = raw.get("y", raw.get("cy", 0.0))
        radius = raw.get("radius", raw.get("r", 1.0))
        ships = raw.get("ships", raw.get("ship_count", 0))
        production = raw.get("production", raw.get("prod", 1))
    else:
        pid, owner, x, y, radius, ships, production = index, -1, 0.0, 0.0, 1.0, 0, 1
    x = float(x)
    y = float(y)
    radius = float(radius)
    orbit_radius = dist_xy(x, y, CENTER_X, CENTER_Y)
    static = orbit_radius + radius >= 50.0 or orbit_radius < EPS or abs(angular_velocity) < EPS
    return {
        "id": int(pid),
        "owner": int(owner) if owner is not None else -1,
        "x": x,
        "y": y,
        "radius": radius,
        "ships": int(max(0, round(float(ships)))),
        "production": float(production),
        "orbit_radius": orbit_radius,
        "orbit_angle": math.atan2(y - CENTER_Y, x - CENTER_X),
        "orbit_speed": 0.0 if static else angular_velocity,
        "static": static,
        "comet": int(pid) in comet_ids,
        "virtual_attack": 0,
        "virtual_reinforce": 0,
        "bad_fire": 0,
    }


def normalise_fleet(raw, index):
    if isinstance(raw, (list, tuple)) and len(raw) >= 7:
        fid, owner, x, y, angle, source_id, ships = raw[:7]
    elif isinstance(raw, dict):
        fid = raw.get("id", index)
        owner = raw.get("owner", raw.get("player", -1))
        x = raw.get("x", 0.0)
        y = raw.get("y", 0.0)
        angle = raw.get("angle", raw.get("direction", 0.0))
        source_id = raw.get("from_planet_id", raw.get("source", -1))
        ships = raw.get("ships", 0)
    else:
        fid, owner, x, y, angle, source_id, ships = index, -1, 0.0, 0.0, 0.0, -1, 0
    return {
        "id": int(fid),
        "owner": int(owner) if owner is not None else -1,
        "x": float(x),
        "y": float(y),
        "angle": float(angle),
        "source": int(source_id) if source_id is not None else -1,
        "ships": int(max(0, round(float(ships)))),
        "target": None,
        "eta": LOOKAHEAD + 1,
    }


def infer_fleet_targets(planets, fleets):
    for fleet in fleets:
        best = None
        speed = fleet_speed(fleet["ships"], 6.0)
        for planet in planets:
            if planet["id"] == fleet["source"]:
                continue
            hit_distance = ray_circle_distance(
                fleet["x"], fleet["y"], fleet["angle"], planet["x"], planet["y"], planet["radius"] + 0.6
            )
            if hit_distance is None:
                continue
            eta = hit_distance / speed
            if best is None or eta < best[0]:
                best = (eta, planet["id"])
        if best:
            fleet["eta"], fleet["target"] = best


def update_memory(state):
    previous = MEMORY.setdefault("turn_by_player", {}).get(state["me"], -1)
    delta = 1 if previous < 0 else max(0, state["turn"] - previous)
    pending_by_player = MEMORY.setdefault("pending_by_player", {})
    pending = pending_by_player.get(state["me"], [])
    if pending and delta > 0:
        owned = {p["id"] for p in state["planets"] if p["owner"] == state["me"]}
        aged = []
        for item in pending:
            eta = max(0.0, item["eta"] - delta)
            if eta <= 0 or item["target"] in owned:
                continue
            copy = dict(item)
            copy["eta"] = eta
            aged.append(copy)
        pending_by_player[state["me"]] = aged
    MEMORY["turn_by_player"][state["me"]] = state["turn"]


def remember_pending(plans, state):
    if not plans:
        return
    pending = MEMORY.setdefault("pending_by_player", {}).setdefault(state["me"], [])
    for plan in plans:
        if plan["reason"] in ("expand", "attack", "steal", "rescue", "reinforce"):
            pending.append(plan)
    if len(pending) > 80:
        del pending[:-80]


def pending_launches(state):
    return MEMORY.setdefault("pending_by_player", {}).get(state["me"], [])


def looks_initial(planets, fleets, me):
    if fleets:
        return False
    active = [p for p in planets if p["owner"] not in (-1, None)]
    owned = [p for p in active if p["owner"] == me]
    return len(owned) == 1 and all(p["ships"] <= 12 for p in active)


# ---------------------------------------------------------------------------
# Strategic phase, budgets, and target choice


def action_limit_for_turn(state):
    owned = len(my_planets(state))
    prod = my_production(state)
    turn = state["turn"]
    if state["crowded"]:
        if turn < 12:
            return 1
        if turn < 35:
            return 2 if owned >= 2 else 1
        if turn < 75:
            return 3 if owned >= 4 or prod >= 18 else 2
        if turn < 130:
            return 5 if prod >= 32 else 4
        if turn < 220:
            return 8
        return 12
    if turn < 10:
        return 1
    if turn < 35:
        return 2 if owned >= 2 else 1
    if turn < 75:
        return 4 if prod >= 16 else 3
    if turn < 130:
        return 7
    return 16


def available_ships(planet, state):
    reserve = 0 if state["turn"] < 75 else 1
    if planet["production"] >= 4 and state["turn"] >= 35:
        reserve += 1
    if state["crowded"] and 45 <= state["turn"] < 95 and planet["production"] >= 4:
        reserve += 1
    if state["turn"] > 120:
        reserve += int(planet["ships"] * (0.06 if not state["crowded"] else 0.10))

    incoming_enemy = 0
    incoming_friend = int(planet.get("virtual_reinforce", 0))
    earliest = LOOKAHEAD
    for fleet in state["fleets"]:
        if fleet["target"] != planet["id"] or fleet["eta"] > LOOKAHEAD:
            continue
        earliest = min(earliest, fleet["eta"])
        if fleet["owner"] == state["me"]:
            incoming_friend += fleet["ships"]
        elif fleet["owner"] != -1:
            incoming_enemy += fleet["ships"]
    local_growth = int(planet["production"] * min(24, earliest))
    reserve += max(0, incoming_enemy - incoming_friend - local_growth + 2)
    return max(0, int(planet["ships"] - reserve))


def best_neutral_capture(source, budget, state):
    best = None
    for target in state["planets"]:
        if target["owner"] != -1 or target["id"] == source["id"]:
            continue
        if should_ignore_comet(target, state):
            continue
        distance = dist_planets(source, target)
        if not neutral_distance_ok(source, target, distance, state):
            continue
        ships, eta = needed_for_target(source, target, budget, state, neutral=True)
        if ships is None or ships > budget:
            continue
        if target["production"] <= 1 and not cheap_bridge_target(target, distance, ships, state):
            continue
        solution = shot_solution(source, target, ships, state)
        if solution is None:
            continue
        score = neutral_score(source, target, ships, eta, distance, state)
        threshold = neutral_score_threshold(target, state)
        if score <= threshold:
            continue
        if best is None or score > best[0]:
            best = (score, target, ships)
    if not best:
        return None
    return best[1], int(best[2]), "expand"


def best_enemy_attack(source, budget, state):
    if not enemy_attack_allowed(state):
        return None
    best = None
    for target in state["planets"]:
        if target["owner"] in (-1, None, state["me"]):
            continue
        distance = dist_planets(source, target)
        if distance > attack_distance_cap(target, state):
            continue
        ships, eta = needed_for_target(source, target, budget, state, neutral=False)
        if ships is None or ships > budget:
            continue
        solution = shot_solution(source, target, ships, state)
        if solution is None:
            continue
        score = enemy_score(source, target, ships, eta, distance, state)
        threshold = neutral_score_threshold(target, state)
        if score <= threshold:
            continue
        if best is None or score > best[0]:
            best = (score, target, ships)
    if not best:
        return None
    return best[1], int(best[2]), "attack"


def best_rescue(source, budget, state):
    best = None
    for target in my_planets(state):
        if target["id"] == source["id"]:
            continue
        need = threatened_amount(target, state) - int(target.get("virtual_reinforce", 0))
        if need <= 0:
            continue
        distance = dist_planets(source, target)
        ships = min(budget, need + 2)
        if ships < MIN_SEND:
            continue
        solution = shot_solution(source, target, ships, state)
        if solution is None:
            continue
        value = target["production"] * 80 + target["ships"] - distance
        if best is None or value > best[0]:
            best = (value, target, ships)
    if not best:
        return None
    return best[1], int(best[2]), "rescue"


def best_reinforce(source, budget, state):
    if budget < MIN_SEND or state["turn"] > 160:
        return None
    best = None
    for target in my_planets(state):
        if target["id"] == source["id"] or target["production"] < 3:
            continue
        distance = dist_planets(source, target)
        if distance > 42:
            continue
        pressure = enemy_pressure_near(target, state)
        frontier = frontier_value(target, state)
        if pressure <= 0 and frontier <= 0:
            continue
        ships = min(budget, max(MIN_SEND, budget // 2))
        solution = shot_solution(source, target, ships, state)
        if solution is None:
            continue
        value = pressure * 2.0 + frontier + target["production"] * 4.0 - distance * 0.2
        if best is None or value > best[0]:
            best = (value, target, ships)
    if not best:
        return None
    return best[1], int(best[2]), "reinforce"


def group_capture_plan(sources, budgets, state, action_limit):
    if action_limit < 2 or state["turn"] < 18:
        return None
    best = None
    for target in state["planets"]:
        if target["owner"] == state["me"]:
            continue
        if target["owner"] != -1 and not enemy_attack_allowed(state):
            continue
        if target["production"] < 4 and target["owner"] == -1:
            continue
        if target["production"] < 3 and target["owner"] != -1:
            continue
        contributors = []
        max_eta = 1.0
        for source in sources:
            cap = budgets.get(source["id"], 0)
            if cap < MIN_SEND:
                continue
            distance = dist_planets(source, target)
            if distance > (60 if state["crowded"] else 74):
                continue
            trial = min(cap, max(MIN_SEND, int(cap)))
            solution = shot_solution(source, target, trial, state)
            if solution is None:
                continue
            angle, eta = solution
            max_eta = max(max_eta, eta)
            contributors.append((distance, source, cap, angle, eta))
        if len(contributors) < 2:
            continue
        needed = required_ships(target, max_eta, state, target["owner"] == -1)
        needed -= committed_to_target(target["id"], state, None)
        needed = int(max(MIN_SEND, needed))
        if sum(c[2] for c in contributors) < needed:
            continue
        contributors.sort(key=lambda x: x[0])
        value = target["production"] * (80 if target["owner"] == -1 else 120) - needed - contributors[0][0]
        if state["crowded"] and target["owner"] != -1 and state["turn"] < 90:
            value -= 150
        if best is None or value > best[0]:
            best = (value, target, needed, contributors)
    if not best or best[0] <= 0:
        return None

    _, target, needed, contributors = best
    actions = []
    plans = []
    remaining = needed
    for _, source, cap, angle, eta in contributors[:min(4, action_limit)]:
        if remaining <= 0:
            break
        send = int(min(cap, max(MIN_SEND, remaining)))
        if send < MIN_SEND:
            continue
        solved = shot_solution(source, target, send, state)
        if solved is None:
            continue
        angle, eta = solved
        actions.append([source["id"], angle, send])
        plans.append(make_plan(source, target, send, eta, "attack" if target["owner"] != -1 else "expand"))
        remaining -= send
    if remaining > 0 or len(actions) < 2:
        return None
    return {"actions": actions, "plans": plans}


# ---------------------------------------------------------------------------
# Scoring and tactical gates


def needed_for_target(source, target, budget, state, neutral):
    trial = max(MIN_SEND, min(budget, max(MIN_SEND, target["ships"] + 2)))
    solution = shot_solution(source, target, trial, state)
    if solution is None:
        return None, None
    _, eta = solution
    needed = required_ships(target, eta, state, neutral)
    needed -= committed_to_target(target["id"], state, None)
    needed = int(max(MIN_SEND, math.ceil(needed)))
    return needed, eta


def required_ships(target, eta, state, neutral):
    if neutral:
        margin = 2 if target["production"] >= 4 else 1
        return target["ships"] + margin
    incoming_friend = existing_fleet_ships(target["id"], state, state["me"], eta + 4)
    incoming_enemy = existing_enemy_reinforce(target, state, eta + 4)
    growth = target["production"] * eta
    return target["ships"] + growth + incoming_enemy - incoming_friend + 3


def neutral_score(source, target, ships, eta, distance, state):
    remaining = max(25.0, state["episode_steps"] - state["turn"] - eta)
    value = target["production"] * remaining
    if target["production"] >= 5:
        value += 360
    elif target["production"] == 4:
        value += 250
    elif target["production"] == 3:
        value += 125
    elif target["production"] == 2 and distance <= 34:
        value += 45
    if distance <= 24:
        value += 110
    elif distance <= 38:
        value += 60
    if source["production"] >= 4:
        value += 55
    if source["production"] >= 3 and target["production"] >= 3:
        value += 35
    if target["comet"]:
        value *= 0.72
    if state["crowded"] and state["turn"] < 80:
        value += 80 if target["production"] >= 4 and distance <= 44 else 0
    cost = ships * 14.0 + eta * 4.0 + distance * 1.45
    return value - cost


def neutral_score_threshold(target, state):
    if target["production"] >= 4 and state["turn"] < 90:
        return -90.0
    if target["production"] == 3 and state["turn"] < 70:
        return -35.0
    return 0.0


def enemy_score(source, target, ships, eta, distance, state):
    my_prod = my_production(state)
    owner_prod = production_for_owner(target["owner"], state)
    value = target["production"] * 170 + target["ships"] * 0.4
    if owner_prod > my_prod:
        value += target["production"] * 75
    if is_weak_enemy(target["owner"], state):
        value += 140
    if distance <= 36:
        value += 45
    if state["crowded"] and state["turn"] < 110 and not is_weak_enemy(target["owner"], state):
        value -= 240
    cost = ships * 16.0 + eta * 6.0 + distance * 2.0
    return value - cost


def neutral_distance_ok(source, target, distance, state):
    if target["production"] >= 4:
        cap = 78 if not state["crowded"] else 64
    elif target["production"] == 3:
        cap = 64 if not state["crowded"] else 52
    elif target["production"] == 2:
        cap = 52 if not state["crowded"] else 42
    else:
        cap = 28
    if source["production"] >= 4:
        cap += 8
    if state["turn"] < 22:
        if target["production"] >= 4:
            cap = min(cap, 58)
        elif target["production"] == 3:
            cap = min(cap, 46)
        else:
            cap = min(cap, 36)
    return distance <= cap


def enemy_attack_allowed(state):
    turn = state["turn"]
    if state["crowded"]:
        return turn >= 82 and (my_production(state) >= 28 or my_planet_count(state) >= 8)
    return turn >= 30 and (my_production(state) >= 12 or my_planet_count(state) >= 3)


def attack_distance_cap(target, state):
    if target["production"] >= 4:
        return 70 if not state["crowded"] else 56
    return 52 if not state["crowded"] else 42


def cheap_bridge_target(target, distance, ships, state):
    return distance <= 22 and ships <= 10 and my_planet_count(state) >= 2


def should_ignore_comet(target, state):
    return target["comet"] and state["turn"] < 70 and target["production"] < 4


def threatened_amount(planet, state):
    incoming_enemy = 0
    incoming_friend = int(planet.get("virtual_reinforce", 0))
    earliest = LOOKAHEAD
    for fleet in state["fleets"]:
        if fleet["target"] != planet["id"] or fleet["eta"] > LOOKAHEAD:
            continue
        earliest = min(earliest, fleet["eta"])
        if fleet["owner"] == state["me"]:
            incoming_friend += fleet["ships"]
        elif fleet["owner"] != -1:
            incoming_enemy += fleet["ships"]
    local = planet["ships"] + int(planet["production"] * min(earliest, 24)) + incoming_friend
    return max(0, incoming_enemy - local + 1)


def enemy_pressure_near(planet, state):
    pressure = threatened_amount(planet, state)
    for other in state["planets"]:
        if other["owner"] in (-1, None, state["me"]):
            continue
        d = dist_planets(planet, other)
        if d <= 34:
            pressure += max(0.0, other["production"] * 2.0 + other["ships"] * 0.04 - d * 0.1)
    return pressure


def frontier_value(planet, state):
    value = 0.0
    for target in state["planets"]:
        if target["owner"] == state["me"]:
            continue
        d = dist_planets(planet, target)
        if d <= 34:
            value += target["production"] * 4.0 - target["ships"] * 0.15 - d * 0.12
    return max(0.0, value)


# ---------------------------------------------------------------------------
# Aiming, geometry, and launch validation


def shot_solution(source, target, ships, state):
    eta = solve_intercept_eta(source, target, ships, state)
    tx, ty = planet_position(target, eta)
    angle = angle_radians(source["x"], source["y"], tx, ty)
    if not valid_shot(source, target, angle, eta, state):
        return None
    return angle, eta


def solve_intercept_eta(source, target, ships, state):
    speed = fleet_speed(ships, state["ship_speed"])
    eta = max(1.0, dist_planets(source, target) / speed)
    for _ in range(8):
        tx, ty = planet_position(target, eta)
        eta = max(1.0, dist_xy(source["x"], source["y"], tx, ty) / speed)
    return eta


def valid_shot(source, target, angle, eta, state):
    sx = source["x"] + math.cos(angle) * (source["radius"] + 0.2)
    sy = source["y"] + math.sin(angle) * (source["radius"] + 0.2)
    tx, ty = planet_position(target, eta)
    sun_clearance = state.get("sun_radius", SUN_RADIUS) + 0.35
    if distance_point_to_segment(CENTER_X, CENTER_Y, sx, sy, tx, ty) <= sun_clearance:
        return False
    if not (0.0 <= tx <= 100.0 and 0.0 <= ty <= 100.0):
        return False
    miss = distance_point_to_segment(tx, ty, sx, sy, sx + math.cos(angle) * 160.0, sy + math.sin(angle) * 160.0)
    return miss <= target["radius"] + 0.75


def planet_position(planet, turn_delta):
    if planet["static"]:
        return planet["x"], planet["y"]
    angle = planet["orbit_angle"] + planet["orbit_speed"] * turn_delta
    return CENTER_X + math.cos(angle) * planet["orbit_radius"], CENTER_Y + math.sin(angle) * planet["orbit_radius"]


def fleet_speed(ships, max_speed=6.0):
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


# ---------------------------------------------------------------------------
# Reservations and accounting


def make_plan(source, target, ships, eta, reason):
    return {
        "source": source["id"],
        "target": target["id"],
        "ships": int(ships),
        "eta": float(eta),
        "reason": reason,
    }


def reserve_target(target, ships, reason):
    if reason in ("expand", "attack", "steal"):
        target["virtual_attack"] = int(target.get("virtual_attack", 0)) + int(ships)
    else:
        target["virtual_reinforce"] = int(target.get("virtual_reinforce", 0)) + int(ships)


def mark_bad_target(target, ships):
    target["bad_fire"] = int(target.get("bad_fire", 0)) + int(ships)


def committed_to_target(target_id, state, owner):
    total = 0
    owner_filter = state["me"] if owner is None else owner
    for item in pending_launches(state):
        if item["target"] == target_id and owner_filter == state["me"]:
            total += int(item["ships"])
    for fleet in state["fleets"]:
        if fleet["target"] == target_id and fleet["owner"] == owner_filter:
            total += int(fleet["ships"])
    return total


def existing_fleet_ships(target_id, state, owner, eta_limit):
    total = 0
    for fleet in state["fleets"]:
        if fleet["target"] == target_id and fleet["owner"] == owner and fleet["eta"] <= eta_limit:
            total += fleet["ships"]
    for item in pending_launches(state):
        if item["target"] == target_id and item["eta"] <= eta_limit:
            total += item["ships"]
    return total


def existing_enemy_reinforce(target, state, eta_limit):
    total = 0
    for fleet in state["fleets"]:
        if fleet["target"] == target["id"] and fleet["owner"] == target["owner"] and fleet["eta"] <= eta_limit:
            total += fleet["ships"]
    return total


def sanitize_actions(actions, plans, state):
    remaining = {p["id"]: max(0, p["ships"] - 1) for p in my_planets(state)}
    clean_actions = []
    clean_plans = []
    plan_index = 0
    for action in actions[:MAX_ACTIONS]:
        if not isinstance(action, (list, tuple)) or len(action) < 3:
            plan_index += 1
            continue
        source_id, angle, ships = action[:3]
        try:
            source_id = int(source_id)
            angle = float(angle) % (2.0 * math.pi)
            ships = int(max(0, round(float(ships))))
        except Exception:
            plan_index += 1
            continue
        if source_id not in remaining:
            plan_index += 1
            continue
        ships = min(ships, remaining[source_id])
        if ships < MIN_SEND:
            plan_index += 1
            continue
        clean_actions.append([source_id, angle, ships])
        if plan_index < len(plans):
            plan = dict(plans[plan_index])
            plan["ships"] = ships
            clean_plans.append(plan)
        remaining[source_id] -= ships
        plan_index += 1
    return clean_actions, clean_plans


# ---------------------------------------------------------------------------
# Basic summaries


def my_planets(state):
    return [p for p in state["planets"] if p["owner"] == state["me"]]


def my_planet_count(state):
    return len(my_planets(state))


def my_production(state):
    return sum(p["production"] for p in my_planets(state))


def production_for_owner(owner, state):
    return sum(p["production"] for p in state["planets"] if p["owner"] == owner)


def total_for_owner(owner, state):
    return sum(p["ships"] + p["production"] * 20 for p in state["planets"] if p["owner"] == owner)


def is_weak_enemy(owner, state):
    if owner in (-1, None, state["me"]):
        return False
    mine = max(1.0, total_for_owner(state["me"], state))
    enemy = total_for_owner(owner, state)
    return enemy < mine * 0.72 or production_for_owner(owner, state) < my_production(state) * 0.65


# ---------------------------------------------------------------------------
# Utility and fallback


def fallback_action(obs, config=None):
    planets = [normalise_planet(p, i, float(first_present(obs, ("angular_velocity",), 0.0)), set()) for i, p in enumerate(as_list(first_present(obs, ("planets",), [])))]
    if not planets:
        return []
    me = int(first_present(obs, ("player", "mark", "player_id"), 0))
    mine = [p for p in planets if p["owner"] == me and p["ships"] > MIN_SEND + 1]
    targets = [p for p in planets if p["owner"] != me]
    if not mine or not targets:
        return []
    source = max(mine, key=lambda p: p["ships"])
    target = min(targets, key=lambda p: dist_planets(source, p) + p["ships"] * 1.5 - p["production"] * 8.0)
    ships = min(source["ships"] - 1, max(MIN_SEND, target["ships"] + 1))
    if ships < MIN_SEND:
        return []
    return [[source["id"], angle_radians(source["x"], source["y"], target["x"], target["y"]), int(ships)]]


def first_present(obj, names, default=None):
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def as_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    return list(value)


def dist_planets(a, b):
    return dist_xy(a["x"], a["y"], b["x"], b["y"])


def dist_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def angle_radians(ax, ay, bx, by):
    return math.atan2(by - ay, bx - ax) % (2.0 * math.pi)


def ray_circle_distance(px, py, angle, cx, cy, radius):
    dx = math.cos(angle)
    dy = math.sin(angle)
    fx = cx - px
    fy = cy - py
    projection = fx * dx + fy * dy
    if projection <= 0:
        return None
    closest_sq = fx * fx + fy * fy - projection * projection
    if closest_sq > radius * radius:
        return None
    offset = math.sqrt(max(0.0, radius * radius - closest_sq))
    distance = projection - offset
    return distance if distance > 0 else projection + offset


def distance_point_to_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= EPS:
        return dist_xy(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return dist_xy(px, py, ax + t * dx, ay + t * dy)


def agent(obs, config=None):
    return _agent_impl(obs, config)
