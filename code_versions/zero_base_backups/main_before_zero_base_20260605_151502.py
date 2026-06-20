"""Orbit Wars Kaggle agent.

This file is intentionally self-contained for Kaggle submission.  The agent is a
surplus dispatcher: keep enough ships home to survive near-term threats, then
turn every extra ship into production, enemy denial, or finishing pressure.
"""

from __future__ import annotations

import math


BOARD_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
MAX_ACTIONS = 12
LOOKAHEAD = 80
MIN_SEND = 3

MEMORY = {
    "turn": -1,
    "turn_by_player": {},
    "enemy_totals": {},
    "last_my_production": 0.0,
    "pending_launches": {},
}


def _agent_impl(obs, config=None):
    """Return a list of `[from_planet_id, angle_radians, ships]` actions."""
    try:
        state = parse_state(obs, config)
        planets = state["planets"]
        if not planets:
            return []

        update_memory(state)
        if state["turn"] <= 0:
            return []
        my_planets = [p for p in planets if p["owner"] == state["me"]]
        if not my_planets:
            return []

        actions = []
        pending_records = []
        budgets = {p["id"]: available_ships(p, state) for p in my_planets}
        sources = sorted(my_planets, key=lambda p: budgets[p["id"]], reverse=True)

        action_limit = action_limit_for_turn(state)
        group_plan = coordinated_group_attack(sources, budgets, state, action_limit)
        if group_plan:
            cleaned = sanitize_actions(group_plan["actions"], state)
            remember_pending_launches(
                cleaned,
                [(source, group_plan["target"], ships, "attack") for source, _, ships in group_plan["contributors"]],
                state,
            )
            return cleaned

        for source in sources:
            if should_hold_for_opening_packet(source, budgets[source["id"]], state):
                continue
            while budgets[source["id"]] >= MIN_SEND and len(actions) < action_limit:
                mission = choose_mission(source, budgets[source["id"]], state)
                if mission is None:
                    break

                target, ships, reason = mission
                if (
                    reason == "attack"
                    and state["turn"] < 15
                    and target["owner"] == -1
                    and target["ships"] <= 8
                    and target["production"] >= 2
                    and dist_planets(source, target) < 32
                ):
                    ships = min(budgets[source["id"]], max(ships + 4, int(target["ships"] + target["production"] * 3)))
                if should_enforce_opening_packet(source, target, ships, budgets[source["id"]], state):
                    desired = desired_opening_packet(source, target, state)
                    if budgets[source["id"]] < desired:
                        break
                    ships = max(ships, desired)
                max_launch = max(0, source["ships"] - 1)
                if max_launch < MIN_SEND:
                    break
                ships = int(max(MIN_SEND, min(ships, budgets[source["id"]], max_launch)))
                if reason != "rescue" and ships < min_packet_for_action(reason, state):
                    break
                if ships < MIN_SEND:
                    break

                angle = intercept_angle(source, target, ships, state["turn"])
                distance = dist_planets(source, target)
                if not safe_launch_path(source, angle, distance, state):
                    break
                actions.append([source["id"], angle, ships])
                pending_records.append((source, target, ships, reason))
                budgets[source["id"]] -= ships

                if reason in ("rescue", "reinforce"):
                    target["virtual_reinforce"] = target.get("virtual_reinforce", 0) + ships
                    if reason == "reinforce":
                        break
                else:
                    target["virtual_attack"] = target.get("virtual_attack", 0) + ships
                    break

        cleaned = sanitize_actions(actions, state)
        remember_pending_launches(cleaned, pending_records, state)
        return cleaned
    except Exception:
        return fallback_action(obs, config)


def parse_state(obs, config):
    explicit_me = first_present(obs, ("player", "mark", "player_id", "id"), None)
    me = int(0 if explicit_me is None else explicit_me)
    raw_planets = first_present(obs, ("planets", "map", "entities"), [])
    raw_fleets = first_present(obs, ("fleets", "ships", "moving_fleets"), [])
    angular_velocity = float(first_present(obs, ("angular_velocity",), 0.0))

    planets = [normalise_planet(p, i, angular_velocity) for i, p in enumerate(as_list(raw_planets))]
    fleets = [normalise_fleet(f, i) for i, f in enumerate(as_list(raw_fleets))]
    infer_fleet_targets(planets, fleets)

    if explicit_me is None and planets and me not in {p["owner"] for p in planets if p["owner"] is not None}:
        owners = sorted({p["owner"] for p in planets if p["owner"] not in (None, -1)})
        if owners:
            me = owners[0]

    turn_raw = first_present(obs, ("step", "turn"), None)
    previous_turn = MEMORY.get("turn_by_player", {}).get(me, -1)
    if turn_raw is None:
        turn = 0 if looks_like_initial_observation(planets, fleets, me) else previous_turn + 1
    else:
        turn = int(turn_raw)
    if turn <= 0 or turn < previous_turn:
        MEMORY["turn_by_player"][me] = -1
        MEMORY.setdefault("pending_launches", {}).pop(me, None)

    players = sorted({p["owner"] for p in planets if p["owner"] not in (None, -1)})
    return {
        "me": me,
        "turn": turn,
        "planets": planets,
        "fleets": fleets,
        "players": players,
        "episode_steps": int(first_present(config, ("episodeSteps", "episode_steps"), 500)),
        "ship_speed": float(first_present(config, ("shipSpeed", "ship_speed"), 6.0)),
        "sun_radius": float(first_present(config, ("sunRadius", "sun_radius"), SUN_RADIUS)),
    }


def looks_like_initial_observation(planets, fleets, me):
    if fleets:
        return False
    active = [p for p in planets if p["owner"] not in (None, -1)]
    owned = [p for p in active if p["owner"] == me]
    return len(owned) == 1 and owned[0]["ships"] <= 12 and all(p["ships"] <= 12 for p in active)

def normalise_planet(raw, index, angular_velocity=0.0):
    if isinstance(raw, (list, tuple)) and len(raw) >= 7:
        raw_id, owner, x, y, radius, ships, production = raw[:7]
        orbit_radius = dist_xy(float(x), float(y), 50.0, 50.0)
        return {
            "id": int(raw_id),
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "ships": int(ships),
            "production": float(production),
            "owner": int(owner),
            "orbit_radius": orbit_radius,
            "orbit_speed": float(angular_velocity) if orbit_radius + float(radius) < 50.0 else 0.0,
            "orbit_angle": math.atan2(float(y) - 50.0, float(x) - 50.0),
            "static": orbit_radius + float(radius) >= 50.0,
        }

    raw_id = first_present(raw, ("id", "planet_id", "index"), index)
    x = float(first_present(raw, ("x", "cx", "pos_x"), 50.0))
    y = float(first_present(raw, ("y", "cy", "pos_y"), 50.0))
    radius = float(first_present(raw, ("radius", "r"), 1.0))
    ships = int(first_present(raw, ("ships", "ship_count", "num_ships", "garrison"), 0))
    production = float(first_present(raw, ("production", "growth", "growth_rate", "ship_production"), 0.0))
    owner = first_present(raw, ("owner", "player", "owner_id"), 0)
    owner = int(owner) if owner is not None else 0
    orbit_radius = float(first_present(raw, ("orbit_radius", "distance", "orbital_radius"), dist_xy(x, y, 50, 50)))
    orbit_speed = float(first_present(raw, ("orbit_speed", "angular_velocity", "omega"), angular_velocity))
    orbit_angle = float(first_present(raw, ("orbit_angle", "theta", "angle"), math.atan2(y - 50, x - 50)))
    static = bool(first_present(raw, ("static", "is_static", "stationary"), abs(orbit_speed) < 1e-9))
    return {
        "id": int(raw_id),
        "x": x,
        "y": y,
        "radius": radius,
        "ships": ships,
        "production": production,
        "owner": owner,
        "orbit_radius": orbit_radius,
        "orbit_speed": orbit_speed,
        "orbit_angle": orbit_angle,
        "static": static,
    }


def normalise_fleet(raw, index):
    if isinstance(raw, (list, tuple)) and len(raw) >= 7:
        raw_id, owner, x, y, angle, source, ships = raw[:7]
        return {
            "id": int(raw_id),
            "owner": int(owner),
            "x": float(x),
            "y": float(y),
            "angle": float(angle),
            "source": int(source),
            "ships": int(ships),
            "target": None,
            "eta": 999.0,
        }

    owner = first_present(raw, ("owner", "player", "owner_id"), None)
    return {
        "id": int(first_present(raw, ("id", "fleet_id", "index"), index)),
        "owner": int(owner) if owner is not None else None,
        "x": float(first_present(raw, ("x", "cx", "pos_x"), 50.0)),
        "y": float(first_present(raw, ("y", "cy", "pos_y"), 50.0)),
        "angle": float(first_present(raw, ("angle", "theta", "direction"), 0.0)),
        "ships": int(first_present(raw, ("ships", "ship_count", "num_ships"), 0)),
        "target": first_present(raw, ("target", "target_id", "destination"), None),
        "source": first_present(raw, ("source", "source_id", "origin"), None),
        "eta": float(first_present(raw, ("eta", "turns_remaining", "arrival"), 999)),
    }


def infer_fleet_targets(planets, fleets):
    for fleet in fleets:
        if fleet["target"] is not None:
            continue
        best = None
        for planet in planets:
            if planet["id"] == fleet["source"]:
                continue
            distance = ray_circle_distance(
                fleet["x"],
                fleet["y"],
                fleet["angle"],
                planet["x"],
                planet["y"],
                planet["radius"],
            )
            if distance is None:
                continue
            eta = distance / fleet_speed(fleet["ships"])
            if best is None or eta < best[0]:
                best = (eta, planet["id"])
        if best is not None:
            fleet["eta"], fleet["target"] = best


def update_memory(state):
    MEMORY["turn"] = state["turn"]
    previous_turn = MEMORY.setdefault("turn_by_player", {}).get(state["me"], -1)
    pending_by_player = MEMORY.setdefault("pending_launches", {})
    pending = pending_by_player.get(state["me"], [])
    delta = 1 if previous_turn < 0 else max(0, state["turn"] - previous_turn)
    if pending and delta > 0:
        owned_ids = {p["id"] for p in state["planets"] if p["owner"] == state["me"]}
        aged = []
        for launch in pending:
            eta = max(0.0, float(launch["eta"]) - delta)
            if eta <= 0 or launch["target"] in owned_ids:
                continue
            updated = dict(launch)
            updated["eta"] = eta
            aged.append(updated)
        pending_by_player[state["me"]] = aged
    MEMORY.setdefault("turn_by_player", {})[state["me"]] = state["turn"]
    totals = {}
    for p in state["planets"]:
        totals[p["owner"]] = totals.get(p["owner"], 0) + p["ships"] + 20 * p["production"]
    MEMORY["enemy_totals"] = {k: v for k, v in totals.items() if k not in (-1, state["me"])}
    MEMORY["last_my_production"] = sum(p["production"] for p in state["planets"] if p["owner"] == state["me"])


def available_ships(planet, state):
    if state["turn"] < 55:
        reserve = 0 if planet["ships"] <= 18 else 1
    elif state["turn"] < 140:
        reserve = max(2, int(planet["ships"] * 0.10))
    else:
        reserve = max(3, int(planet["ships"] * 0.18))
    incoming_enemy = 0
    incoming_friendly = planet.get("virtual_reinforce", 0)
    for fleet in state["fleets"]:
        if fleet["target"] != planet["id"] or fleet["eta"] > LOOKAHEAD:
            continue
        if fleet["owner"] == state["me"]:
            incoming_friendly += fleet["ships"]
        else:
            incoming_enemy += fleet["ships"]

    produced_before_threat = int(planet["production"] * min(LOOKAHEAD, 24))
    danger = max(0, incoming_enemy - incoming_friendly - produced_before_threat)
    reserve += danger
    if crowded_mode(state):
        if state["turn"] < 90:
            reserve += 1 if planet["production"] >= 2 else 0
        elif state["turn"] < 150:
            reserve += 2
    if state["turn"] >= 60:
        reserve += 2
    return max(0, planet["ships"] - reserve)


def coordinated_group_attack(sources, budgets, state, action_limit):
    if action_limit < 2:
        return None
    my_production = sum(p["production"] for p in state["planets"] if p["owner"] == state["me"])
    if state["turn"] < 80 or my_production < 40:
        return None

    best = None
    for target in state["planets"]:
        if target["owner"] == state["me"]:
            continue
        if target["production"] < 4:
            continue
        if avoid_early_enemy_attack(target, state):
            continue
        if target.get("virtual_attack", 0):
            continue

        contributors = []
        max_eta = 1.0
        for source in sources:
            budget = budgets.get(source["id"], 0)
            if budget < coordinated_min_packet(state):
                continue
            distance = dist_planets(source, target)
            if distance > 72:
                continue
            ships = min(budget, source["ships"] if state["turn"] < 60 else source["ships"] - 1)
            if ships < coordinated_min_packet(state):
                continue
            angle = intercept_angle(source, target, ships, state["turn"])
            if not safe_launch_path(source, angle, distance, state):
                continue
            eta = solve_intercept_eta(source, target, ships)
            max_eta = max(max_eta, eta)
            contributors.append((distance, source, ships, angle))

        if len(contributors) < 2:
            continue
        future_growth = 0 if target["owner"] in (-1, None) else target["production"] * max_eta
        needed = int(math.ceil(target["ships"] + future_growth + 3))
        total_budget = sum(item[2] for item in contributors)
        if total_budget < needed:
            continue

        contributors.sort(key=lambda item: item[0])
        value = target["production"] * 28.0
        if target["owner"] not in (-1, None):
            value *= 1.7
        value += max(0.0, 80.0 - contributors[0][0])
        value /= max(1.0, needed)
        if best is None or value > best[0]:
            best = (value, target, needed, contributors)

    if best is None:
        return None

    _, target, needed, contributors = best
    actions = []
    used = []
    remaining = needed
    for _, source, capacity, angle in contributors[:min(action_limit, 3)]:
        if remaining <= 0:
            break
        ships = min(capacity, max(coordinated_min_packet(state), remaining))
        actions.append([source["id"], angle, int(ships)])
        used.append((source, angle, int(ships)))
        remaining -= ships
    if remaining > 0 or len(actions) < 2:
        return None
    return {"actions": actions, "contributors": used, "target": target}

def action_limit_for_turn(state):
    mine = [p for p in state["planets"] if p["owner"] == state["me"]]
    my_production = sum(p["production"] for p in mine)
    owned = len(mine)
    crowded = crowded_mode(state)
    if crowded:
        if state["turn"] < 24:
            return 1
        if state["turn"] < 60:
            return 2 if owned >= 4 or my_production >= 14 else 1
        if state["turn"] < 120:
            return 3 if owned >= 6 and my_production >= 28 else 2
        if state["turn"] < 220:
            return 4 if owned >= 8 and my_production >= 48 else 3
        return 5 if owned >= 10 and my_production >= 64 else 4
    if state["turn"] < 18:
        return 1
    if state["turn"] < 45:
        return 2 if owned >= 2 or my_production >= 10 else 1
    if state["turn"] < 75:
        return 3 if owned >= 5 or my_production >= 24 else 2
    if state["turn"] < 130:
        return 4 if my_production >= 45 else 3
    return MAX_ACTIONS
def choose_mission(source, budget, state):
    rescue = best_rescue(source, budget, state)
    if rescue:
        return rescue
    if state["turn"] < 30 and len([p for p in state["planets"] if p["owner"] == state["me"]]) <= 1 and has_pending_opening_launch(state):
        return None

    chain = bowwow_neutral_chain(source, budget, state)
    if chain:
        return chain

    reinforce = best_reinforce(source, budget, state)
    candidates = []
    for target in state["planets"]:
        if target["id"] == source["id"] or target["owner"] == state["me"]:
            continue
        if avoid_early_enemy_attack(target, state):
            continue
        distance = dist_planets(source, target)
        ships = required_attack_ships(source, target, distance, state)
        if ships < MIN_SEND:
            continue
        if ships > budget:
            continue
        if should_skip_crowded_attack(source, target, ships, distance, state):
            continue
        angle = intercept_angle(source, target, ships, 0)
        if not safe_launch_path(source, angle, distance, state):
            continue
        score = target_score(source, target, ships, distance, state)
        if score > 0:
            candidates.append((score, target, ships))

    if not candidates:
        return reinforce
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, target, ships = candidates[0]
    return target, ships, "attack"

def bowwow_neutral_chain(source, budget, state):
    if state["turn"] > 92 or budget < MIN_SEND:
        return None
    mine = [p for p in state["planets"] if p["owner"] == state["me"]]
    owned = len(mine)
    crowded = crowded_mode(state)
    if owned <= 1 and state["turn"] < 4:
        return None
    my_production = sum(p["production"] for p in mine)
    if owned >= 7 and my_production >= 32:
        return None

    best = None
    max_distance = 46 if state["turn"] < 32 else 56
    if crowded:
        max_distance = 32 if state["turn"] < 36 else 38
    reserve = early_source_reserve(source, state)
    for target in state["planets"]:
        if target["owner"] != -1 or target["id"] == source["id"]:
            continue
        distance = dist_planets(source, target)
        if distance > max_distance:
            continue
        if crowded:
            if target["production"] <= 1:
                continue
            if target["production"] == 2 and (distance > 22 or target["ships"] > 10):
                continue
            if target["production"] <= 3 and distance > 30:
                continue
        elif owned <= 1 and target["production"] <= 2 and distance > 34:
            continue
        if target["production"] < 2 and not (distance <= 28 and target["ships"] <= 12 and owned >= 2):
            continue
        if crowded and target["production"] < 4 and target["ships"] >= 18:
            continue
        margin = 2 if target["production"] >= 4 else 1
        if distance > 40:
            margin += 1
        committed = committed_friendly_attack_ships(target["id"], state, None) + int(target.get("virtual_attack", 0))
        if committed >= target["ships"] + margin:
            continue
        ships = int(max(MIN_SEND, math.ceil(target["ships"] + margin - committed)))
        if ships > budget or ships > source["ships"] - reserve:
            continue
        angle = intercept_angle(source, target, ships, state["turn"])
        if not safe_launch_path(source, angle, distance, state):
            continue
        eta = solve_intercept_eta(source, target, ships)
        value = target["production"] * 34.0 - ships * 1.20 - distance * 0.28 - eta * (1.45 if crowded else 0.85)
        if target["production"] >= 4:
            value += 18.0
        if target["ships"] <= 12:
            value += 8.0
        if distance <= 30:
            value += 9.0
        if state["turn"] < 35:
            value += 6.0
        if crowded and target["production"] == 2:
            value -= 8.0
        if best is None or value > best[0]:
            best = (value, target, ships)
    if best is None or best[0] <= 0:
        return None
    return best[1], best[2], "expand"


def early_source_reserve(source, state):
    owned = len([p for p in state["planets"] if p["owner"] == state["me"]])
    if state["turn"] < 45:
        return 0 if owned <= 1 else max(1, int(source["production"] * 1.2))
    if state["turn"] < 90:
        return max(2, int(source["production"] * 1.5))
    return 2

def avoid_early_enemy_attack(target, state):
    if target["owner"] in (-1, None, state["me"]):
        return False
    my_production = sum(p["production"] for p in state["planets"] if p["owner"] == state["me"])
    if crowded_mode(state):
        return state["turn"] < 95 and my_production < 36
    return state["turn"] < 70 and my_production < 30


def should_skip_crowded_attack(source, target, ships, distance, state):
    if not crowded_mode(state):
        return False

    mine = [p for p in state["planets"] if p["owner"] == state["me"]]
    owned = len(mine)
    my_production = sum(p["production"] for p in mine)
    owner = target["owner"]

    if owner in (-1, None):
        if state["turn"] < 55:
            return False
        if target["production"] >= 4:
            return False
        return not (distance <= 22 and target["ships"] <= 10 and owned >= 4)

    if is_weak_enemy(owner) and distance <= 28 and target["production"] >= 4 and ships >= 10:
        return False
    if state["turn"] < 40:
        return target["production"] <= 3 and distance > 24
    if my_production < 20 or owned < 6:
        return target["production"] < 6 or distance > 30
    if target["production"] <= 4 and distance > 26:
        return True
    if target["production"] <= 6 and ships < 12:
        return True
    if ships > source["ships"] * 0.62 and target["production"] <= 6:
        return True
    return False
def coordinated_min_packet(state):
    return 12 if state["turn"] < 130 else 10


def min_packet_for_action(reason, state):
    if reason in ("rescue", "expand"):
        return MIN_SEND
    if crowded_mode(state):
        if state["turn"] < 80:
            return 12
        if state["turn"] < 180:
            return 10
        return 8
    if state["turn"] < 25:
        return early_attack_floor(state)
    if state["turn"] < 80:
        return 10
    if state["turn"] < 140:
        return 8
    return MIN_SEND
def early_attack_floor(state):
    owned = len([p for p in state["planets"] if p["owner"] == state["me"]])
    return 10 if owned < 4 else 8


def should_enforce_opening_packet(source, target, ships, budget, state):
    if state["turn"] > 15:
        return False
    if target["owner"] != -1:
        return False
    if len([p for p in state["planets"] if p["owner"] == state["me"]]) > 1:
        return False
    return ships < desired_opening_packet(source, target, state)

def should_hold_for_opening_packet(source, budget, state):
    if state["turn"] > 15:
        return False
    if len([p for p in state["planets"] if p["owner"] == state["me"]]) > 1:
        return False
    if has_pending_opening_launch(state):
        return True
    target = best_opening_target(source, state)
    if target is None:
        return False
    desired = desired_opening_packet(source, target, state)
    if desired >= budget + 3:
        return False
    return budget < desired


def best_opening_target(source, state):
    best = None
    crowded = crowded_mode(state)
    for target in state["planets"]:
        if target["owner"] != -1:
            continue
        distance = dist_planets(source, target)
        if crowded:
            if target["production"] <= 1 and distance > 18:
                continue
            if target["production"] == 2 and distance > 30:
                continue
            if target["production"] >= 3 and distance > 40:
                continue
        else:
            if target["production"] <= 2 and distance > 42:
                continue
            if target["production"] == 3 and distance > 50:
                continue
            if target["production"] >= 4 and distance > 58:
                continue
        ships = max(MIN_SEND, target["ships"] + (2 if target["production"] >= 4 else 1))
        eta = travel_turns(distance, ships)
        if crowded and eta > 20:
            continue
        value = target["production"] * 18.0 - target["ships"] * 1.4 - distance * 0.45 - eta * 2.1
        if target["production"] >= 4:
            value += 14.0
        elif target["production"] == 3:
            value += 6.0
        if distance <= 32:
            value += 8.0
        if crowded and distance <= 22:
            value += 8.0
        if best is None or value > best[0]:
            best = (value, target)
    return best[1] if best is not None else None


def desired_opening_packet(source, target, state):
    margin = 2 if target["production"] >= 4 else 1
    if dist_planets(source, target) > 42:
        margin += 1
    base = target["ships"] + margin
    production_floor = 8 + int(source["production"] * 2)
    if target["production"] >= 4:
        production_floor += 2
    if dist_planets(source, target) > 42:
        production_floor += 2
    return int(max(MIN_SEND, base, production_floor))
def best_reinforce(source, budget, state):
    crowded = crowded_mode(state)
    if budget < MIN_SEND:
        return None
    if not crowded and state["turn"] > 140:
        return None
    if crowded and state["turn"] > 240:
        return None
    best = None
    distance_limit = 40 if crowded and state["turn"] < 160 else 34
    for target in state["planets"]:
        if target["owner"] != state["me"] or target["id"] == source["id"]:
            continue
        distance = dist_planets(source, target)
        if distance > distance_limit:
            continue
        nearby_value = 0.0
        for candidate in state["planets"]:
            if candidate["owner"] == state["me"]:
                continue
            d = dist_planets(target, candidate)
            if d > 30:
                continue
            capture_cost = max(1.0, candidate["ships"] + (0 if candidate["owner"] == -1 else candidate["production"] * travel_turns(d, max(3, budget))))
            nearby_value += (candidate["production"] * 18.0 + max(0, 18.0 - d)) / capture_cost
        if nearby_value <= 0:
            continue
        value = nearby_value + target["production"] * 0.8 - distance * 0.08 - target.get("virtual_reinforce", 0) * 0.18
        if best is None or value > best[0]:
            best = (value, target)
    if best is None:
        return None
    ships = min(budget, max(MIN_SEND, budget // 2 if state["turn"] < 45 else budget))
    return best[1], int(ships), "reinforce"


def best_rescue(source, budget, state):
    best = None
    for target in state["planets"]:
        if target["owner"] != state["me"] or target["id"] == source["id"]:
            continue
        need = threatened_by(target, state) - int(target.get("virtual_reinforce", 0))
        if need <= 0:
            continue
        distance = dist_planets(source, target)
        ships = min(budget, need + 2)
        if ships >= MIN_SEND:
            value = target["production"] * 12 + target["ships"] - distance * 0.15
            if best is None or value > best[0]:
                best = (value, target, ships)
    if best is None:
        return None
    return best[1], int(best[2]), "rescue"


def threatened_by(planet, state):
    incoming_enemy = 0
    incoming_friendly = planet.get("virtual_reinforce", 0)
    earliest = LOOKAHEAD
    for fleet in state["fleets"]:
        if fleet["target"] != planet["id"] or fleet["eta"] > LOOKAHEAD:
            continue
        earliest = min(earliest, fleet["eta"])
        if fleet["owner"] == state["me"]:
            incoming_friendly += fleet["ships"]
        else:
            incoming_enemy += fleet["ships"]
    local = planet["ships"] + int(planet["production"] * earliest) + incoming_friendly
    return max(0, incoming_enemy - local + 1)


def has_pending_opening_launch(state):
    return any(
        launch["kind"] in ("expand", "attack") and launch["eta"] > 0
        for launch in MEMORY.get("pending_launches", {}).get(state["me"], [])
    )


def committed_friendly_attack_ships(target_id, state, eta_limit=None):
    total = 0
    for launch in MEMORY.get("pending_launches", {}).get(state["me"], []):
        if launch["target"] != target_id or launch["kind"] not in ("expand", "attack"):
            continue
        if eta_limit is not None and launch["eta"] > eta_limit:
            continue
        total += launch["ships"]
    return total


def remember_pending_launches(actions, pending_records, state):
    if not actions or not pending_records:
        return
    launches = MEMORY.setdefault("pending_launches", {}).setdefault(state["me"], [])
    for action, (source, target, _, reason) in zip(actions, pending_records):
        ships = int(action[2])
        launches.append({
            "source": source["id"],
            "target": target["id"],
            "ships": ships,
            "eta": solve_intercept_eta(source, target, ships),
            "kind": "reinforce" if reason in ("rescue", "reinforce") else reason,
        })


def required_attack_ships(source, target, distance, state):
    eta = solve_intercept_eta(source, target, max(MIN_SEND, min(source["ships"], 80)))
    future_growth = 0 if target["owner"] in (-1, None) else target["production"] * eta
    contest = int(target.get("virtual_attack", 0))
    observed_friendly = sum(
        f["ships"] for f in state["fleets"]
        if f["target"] == target["id"] and f["owner"] == state["me"] and f["eta"] < eta + 5
    )
    remembered_friendly = committed_friendly_attack_ships(target["id"], state, eta + 5)
    incoming_friendly = max(observed_friendly, remembered_friendly)
    incoming_enemy = sum(
        f["ships"] for f in state["fleets"]
        if f["target"] == target["id"] and f["owner"] != state["me"] and f["eta"] < eta + 5
    )
    margin = 1 if target["owner"] in (-1, None) else 2
    needed = target["ships"] + future_growth + incoming_enemy - incoming_friendly - contest + margin
    if needed <= 0:
        return 0
    return int(max(MIN_SEND, math.ceil(needed)))


def target_score(source, target, ships, distance, state):
    eta = solve_intercept_eta(source, target, ships)
    remaining = max(20, state["episode_steps"] - state["turn"] - eta)
    production_value = target["production"] * remaining
    deny_value = production_value if target["owner"] not in (-1, None, state["me"]) else 0
    base = production_value + 0.75 * deny_value + target["ships"] * 0.12

    owner = target["owner"]
    crowded = crowded_mode(state)
    my_production = sum(p["production"] for p in state["planets"] if p["owner"] == state["me"])
    multiplier = 1.0
    if owner not in (-1, None, state["me"]):
        multiplier *= 1.85
        if is_weak_enemy(owner):
            multiplier *= 1.35
    else:
        multiplier *= 1.18 if target["static"] else 1.0

    if target.get("virtual_attack", 0):
        multiplier *= 0.72
    if state["turn"] < 80 and owner in (-1, None):
        multiplier *= 1.65
        if distance <= 36:
            multiplier *= 1.25
        if target["ships"] <= source["ships"] - 1:
            multiplier *= 1.35
    if crowded:
        if owner not in (-1, None, state["me"]):
            multiplier *= 0.78
            if state["turn"] < 120 and my_production < 24:
                multiplier *= 0.55
        elif state["turn"] < 90:
            if target["production"] <= 1:
                multiplier *= 0.35
            elif target["production"] == 2 and distance > 26:
                multiplier *= 0.62
            if distance > 42:
                multiplier *= 0.58
    if avoid_early_enemy_attack(target, state):
        multiplier *= 0.25
    if ships > source["ships"] * 0.65:
        multiplier *= 0.82

    cost = ships + eta * 0.65 + distance * 0.08 + 1.0
    return multiplier * base / cost


def crowded_mode(state):
    return len(state["players"]) >= 3


def is_weak_enemy(owner):
    totals = MEMORY.get("enemy_totals", {})
    if not totals or owner not in totals:
        return False
    return totals[owner] <= min(totals.values()) * 1.15


def intercept_angle(source, target, ships, turn):
    eta = solve_intercept_eta(source, target, ships)
    tx, ty = planet_position(target, eta)
    return angle_radians(source["x"], source["y"], tx, ty)


def solve_intercept_eta(source, target, ships):
    eta = travel_turns(dist_planets(source, target), ships)
    if target["static"] or abs(target["orbit_speed"]) < 1e-9:
        return eta
    for _ in range(6):
        tx, ty = planet_position(target, eta)
        next_eta = travel_turns(dist_xy(source["x"], source["y"], tx, ty), ships)
        if abs(next_eta - eta) < 0.05:
            return next_eta
        eta = next_eta
    return eta


def planet_position(planet, turn):
    if planet["static"] or abs(planet["orbit_speed"]) < 1e-9:
        return planet["x"], planet["y"]
    angle = planet["orbit_angle"] + planet["orbit_speed"] * turn
    return (
        BOARD_CENTER[0] + math.cos(angle) * planet["orbit_radius"],
        BOARD_CENTER[1] + math.sin(angle) * planet["orbit_radius"],
    )


def avoid_sun_angle(source, target, angle):
    sx, sy = source["x"], source["y"]
    tx, ty = target["x"], target["y"]
    if distance_point_to_segment(BOARD_CENTER[0], BOARD_CENTER[1], sx, sy, tx, ty) > SUN_RADIUS + 1.7:
        return angle
    side = 1 if cross(sx - 50, sy - 50, tx - 50, ty - 50) >= 0 else -1
    return (angle + side * math.radians(15.0)) % (2.0 * math.pi)


def safe_launch_path(source, angle, distance, state):
    sx = source["x"] + math.cos(angle) * (source["radius"] + 0.15)
    sy = source["y"] + math.sin(angle) * (source["radius"] + 0.15)
    ex = sx + math.cos(angle) * min(distance + 4.0, 140.0)
    ey = sy + math.sin(angle) * min(distance + 4.0, 140.0)
    clearance = state.get("sun_radius", SUN_RADIUS) + 0.4
    return distance_point_to_segment(BOARD_CENTER[0], BOARD_CENTER[1], sx, sy, ex, ey) > clearance


def fleet_speed(ships):
    ships = max(1, ships)
    return 1.0 + 5.0 * (math.log(ships) / math.log(1000.0)) ** 1.5


def travel_turns(distance, ships):
    return max(1.0, distance / fleet_speed(ships))


def sanitize_actions(actions, state):
    cleaned = []
    remaining = {p["id"]: max(0, int(p["ships"] - 1)) for p in state["planets"] if p["owner"] == state["me"]}
    for action in actions[:MAX_ACTIONS]:
        if not isinstance(action, (list, tuple)) or len(action) < 3:
            continue
        source_id, angle, ships = action[:3]
        if source_id not in remaining:
            continue
        try:
            angle = float(angle) % (2.0 * math.pi)
            ships = int(ships)
        except Exception:
            continue
        ships = min(ships, remaining[source_id])
        if ships < MIN_SEND:
            continue
        cleaned.append([int(source_id), angle, ships])
        remaining[source_id] -= ships
    return cleaned

def fallback_action(obs, config):
    planets = [normalise_planet(p, i) for i, p in enumerate(as_list(first_present(obs, ("planets",), [])))]
    if not planets:
        return []
    me = int(first_present(obs, ("player", "mark", "player_id"), planets[0]["owner"]))
    mine = [p for p in planets if p["owner"] == me and p["ships"] > 8]
    targets = [p for p in planets if p["owner"] != me]
    if not mine or not targets:
        return []
    source = max(mine, key=lambda p: p["ships"])
    target = min(targets, key=lambda p: dist_planets(source, p) + p["ships"] * 2)
    ships = max(MIN_SEND, source["ships"] // 3)
    return [[source["id"], angle_radians(source["x"], source["y"], target["x"], target["y"]), ships]]


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
    radius_sq = radius * radius
    if closest_sq > radius_sq:
        return None
    offset = math.sqrt(max(0.0, radius_sq - closest_sq))
    distance = projection - offset
    return distance if distance > 0 else projection + offset


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def distance_point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return dist_xy(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return dist_xy(px, py, ax + t * dx, ay + t * dy)


def agent(obs, config=None):
    return _agent_impl(obs, config)
