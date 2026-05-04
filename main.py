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
    "enemy_totals": {},
    "last_my_production": 0.0,
}


def _agent_impl(obs, config=None):
    """Return a list of `[from_planet_id, angle_radians, ships]` actions."""
    try:
        state = parse_state(obs, config)
        planets = state["planets"]
        if not planets:
            return []

        update_memory(state)
        my_planets = [p for p in planets if p["owner"] == state["me"]]
        if not my_planets:
            return []

        actions = []
        budgets = {p["id"]: available_ships(p, state) for p in my_planets}
        sources = sorted(my_planets, key=lambda p: budgets[p["id"]], reverse=True)

        for source in sources:
            if should_hold_for_opening_packet(source, budgets[source["id"]], state):
                continue
            while budgets[source["id"]] >= MIN_SEND and len(actions) < MAX_ACTIONS:
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
                    ships = budgets[source["id"]]
                max_launch = source["ships"] if state["turn"] < 60 else source["ships"] - 1
                ships = int(max(MIN_SEND, min(ships, budgets[source["id"]], max_launch)))
                if ships < MIN_SEND:
                    break

                angle = intercept_angle(source, target, ships, state["turn"])
                angle = avoid_sun_angle(source, target, angle)
                actions.append([source["id"], angle, ships])
                budgets[source["id"]] -= ships

                if reason in ("rescue", "reinforce"):
                    target["virtual_reinforce"] = target.get("virtual_reinforce", 0) + ships
                    if reason == "reinforce":
                        break
                else:
                    target["virtual_attack"] = target.get("virtual_attack", 0) + ships
                    break

        return actions
    except Exception:
        return fallback_action(obs, config)


def parse_state(obs, config):
    me = int(first_present(obs, ("player", "mark", "player_id", "id"), 0))
    turn = int(first_present(obs, ("step", "turn"), MEMORY["turn"] + 1))
    raw_planets = first_present(obs, ("planets", "map", "entities"), [])
    raw_fleets = first_present(obs, ("fleets", "ships", "moving_fleets"), [])
    angular_velocity = float(first_present(obs, ("angular_velocity",), 0.0))

    planets = [normalise_planet(p, i, angular_velocity) for i, p in enumerate(as_list(raw_planets))]
    fleets = [normalise_fleet(f, i) for i, f in enumerate(as_list(raw_fleets))]
    infer_fleet_targets(planets, fleets)

    if planets and me not in {p["owner"] for p in planets if p["owner"] is not None}:
        owners = sorted({p["owner"] for p in planets if p["owner"] not in (None, -1)})
        if owners:
            me = owners[0]

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
    if state["turn"] >= 60:
        reserve += 2
    return max(0, planet["ships"] - reserve)


def choose_mission(source, budget, state):
    rescue = best_rescue(source, budget, state)
    if rescue:
        return rescue

    candidates = []
    for target in state["planets"]:
        if target["id"] == source["id"] or target["owner"] == state["me"]:
            continue
        distance = dist_planets(source, target)
        ships = required_attack_ships(source, target, distance, state)
        if ships > budget:
            continue
        angle = intercept_angle(source, target, ships, 0)
        if not safe_launch_path(source, angle, distance, state):
            continue
        score = target_score(source, target, ships, distance, state)
        if score > 0:
            candidates.append((score, target, ships))

    if not candidates:
        return best_reinforce(source, budget, state)
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, target, ships = candidates[0]
    return target, ships, "attack"


def should_hold_for_opening_packet(source, budget, state):
    if state["turn"] > 4 or source["production"] < 4 or source["ships"] >= 18:
        return False
    if len([p for p in state["planets"] if p["owner"] == state["me"]]) > 1:
        return False
    for target in state["planets"]:
        if target["owner"] != -1:
            continue
        distance = dist_planets(source, target)
        if distance > 36 or not (6 <= target["ships"] <= 10) or target["production"] < 3:
            continue
        ships_soon = source["ships"] + source["production"] * max(0, 5 - state["turn"])
        if ships_soon >= 18 and budget < 16:
            return True
    return False


def best_reinforce(source, budget, state):
    if state["turn"] > 140 or budget < MIN_SEND:
        return None
    best = None
    for target in state["planets"]:
        if target["owner"] != state["me"] or target["id"] == source["id"]:
            continue
        distance = dist_planets(source, target)
        if distance > 34:
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


def required_attack_ships(source, target, distance, state):
    eta = travel_turns(distance, max(MIN_SEND, min(source["ships"], 80)))
    future_growth = 0 if target["owner"] in (-1, None) else target["production"] * eta
    contest = int(target.get("virtual_attack", 0))
    incoming_friendly = sum(
        f["ships"] for f in state["fleets"]
        if f["target"] == target["id"] and f["owner"] == state["me"] and f["eta"] < eta + 5
    )
    incoming_enemy = sum(
        f["ships"] for f in state["fleets"]
        if f["target"] == target["id"] and f["owner"] != state["me"] and f["eta"] < eta + 5
    )
    margin = 1 if target["owner"] in (-1, None) else 2
    needed = target["ships"] + future_growth + incoming_enemy - incoming_friendly - contest + margin
    return int(max(MIN_SEND, math.ceil(needed)))


def target_score(source, target, ships, distance, state):
    eta = travel_turns(distance, ships)
    remaining = max(20, state["episode_steps"] - state["turn"] - eta)
    production_value = target["production"] * remaining
    deny_value = production_value if target["owner"] not in (-1, None, state["me"]) else 0
    base = production_value + 0.75 * deny_value + target["ships"] * 0.12

    owner = target["owner"]
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
        if target["ships"] <= source["ships"] - 1:
            multiplier *= 1.35
    if ships > source["ships"] * 0.65:
        multiplier *= 0.82

    cost = ships + eta * 0.65 + distance * 0.08 + 1.0
    return multiplier * base / cost


def is_weak_enemy(owner):
    totals = MEMORY.get("enemy_totals", {})
    if not totals or owner not in totals:
        return False
    return totals[owner] <= min(totals.values()) * 1.15


def intercept_angle(source, target, ships, turn):
    eta = travel_turns(dist_planets(source, target), ships)
    tx, ty = planet_position(target, eta)
    return angle_radians(source["x"], source["y"], tx, ty)


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
