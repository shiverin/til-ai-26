# TIL-26 Surprise — Strategic Plan

> Generated from a full read of the engine, rules, and all templates.
> Engine code is the source of truth; all numbers reference `server/src/engine/constants.py`.

---

## Key Discoveries

1. **The local 19 "random" opponents never build production buildings**, so they never field a single unit. Local PASS is trivial. Stage 3 is a true free-for-all against other teams' agents — build for hostile, competent opponents.
2. **Survival = co-win.** Every surviving player at turn 300 wins equally. Optimal posture: maximum economy + defense, aggression only to remove concrete threats.
3. **Several engine quirks** (confirmed in `turn_processor.py`): intermediate path tiles are never occupancy-checked (only the destination); attacks fire pre-move; dead units are removed before movement resolves; splash ignores all treaties/ownership; production queue has no slot limit but spawn-or-lose-gold semantics.

---

## A. Game Analysis

### A1. Dominant Strategy

**"Fortified compound-interest economy"**:

- A Mine (200g, 20g/turn) ordered at turn T repays by T+12, then prints ~20g/turn for the remaining game. Early Mines are worth thousands of gold of lifetime value.
- **Redundant Bases are extra lives.** Elimination = losing the last *completed* Base. 300g Bases planted in quiet corners mean an attacker must find and kill two or three 300-HP buildings to eliminate you.
- Military is **reactive and threshold-triggered**: a small standing garrison plus a war chest. Gold is the most flexible unit — it converts to Infantry in 1 turn once a Barracks exists.

### A2. Economy Breakeven

| Building | Cost | Build | Income | Breakeven | Verdict |
|---|---|---|---|---|---|
| Mine | 200g | 2t | 20g/t | **T+12** | Spam early, always |
| Mine on rich | 200g | 2t | 50g/t | **T+6** | Highest priority once scouted |
| Base | 300g | 5t | 10g/t | T+35 | Bad income; **buy as a life + anchor + vision(+3)** |
| Base on rich | 300g | 5t | 50g/t | **T+11** | Excellent — life *and* income |
| Barracks | 100g | 2t | — | n/a | Buy once early (defense enabler) |
| Factory | 300g | 3t | — | n/a | Mid-game for Artillery defense |
| Airbase | 500g | 3t | — | n/a | Late, only with surplus |

Target income curve: ~30g/t by turn 3, ~70g/t by turn 12, 150g+/t by turn 40.

### A3. Unit Cost-Effectiveness

| Unit | Cost | atk/gold | HP/gold | Notes |
|---|---|---|---|---|
| Infantry | 50g | **0.60** | **2.00** | Best in class. 4 Infantry (200g) = 120 atk + 400 HP vs a Tank's 60/200. Cannot enter difficult terrain (Move 1 < cost 2). |
| Tank | 200g | 0.30 | 1.00 | Mobility (2) for pursuit/raid only. |
| Artillery | 200g | 0.30 (+splash) | 0.25 | 50 HP — glass cannon. Range 3 behind a wall is the best defensive damage in the game. |
| Fighter | 300g | 0.17 | 0.83 | Move 3 / range 2 → kites melee forever. Ignores elevation LOS. |
| Bomber | 350g | **0.57 vs buildings** | 0.43 | ×4 vs buildings (200 dmg/turn). Two Bombers one-shot a full-HP Base (400 ≥ 300). The finisher. |
| Scout | 100g | — | — | Vision 5, Move 3, **invisible in concealment** — permanent hidden watchtower. |
| Medic | 100g | — | — | 20 HP/turn to all adjacent ground units, stacks. Turns an Infantry wall into a regenerating wall. |

### A4. Terrain

- **Elevated**: +25% attack. Station defenders here; ridges block enemy LOS.
- **Difficult**: Infantry-proof corridors (cost 2, Infantry budget 1). A Base pocket behind difficult terrain is naturally defensible.
- **Concealment**: Park Scouts as invisible sentinels on approach corridors. −1 vision-into penalty; Fighters/Bombers see through.
- **Rich resource**: 50g/t flat for Base or Mine. ~10% of map. Scout and claim these first.

### A5. Phase-Ordering & Engine Exploits

All verified in `server/src/engine/turn_processor.py`:

1. **Attacks fire pre-move, then movement** → shoot-and-scoot every ranged unit every turn. Fighter kiting is unbeatable by range-1 units.
2. **Dead units removed before moves** (lines 152–157 before 159) → focus-fire kills a blocker; a unit can step into its tile the same turn.
3. **No overkill prevention, simultaneous damage pool** → assign exactly-lethal damage per target, spread surplus to second targets.
4. **Intermediate path tiles never occupancy-checked** — only the destination (lines 167–201). Units walk through enemies. Pathfinding only needs the *destination* free.
5. **Walling works anyway**: a melee unit must *end* adjacent to attack. Fill all 6 neighbours of a Base with own buildings/units → only range ≥ 2 can hurt it.
6. **Units before buildings** → park a unit on the enemy's expansion tile to silently void their Base build; vacate-and-build on your own tile in one turn.
7. **Attacks need no vision** → Artillery blind-fires splash at suspected areas; also expect blind fire on our positions.
8. **Splash ignores treaties and ownership** (lines 114–123) → treaty-legal to splash an ally by targeting an adjacent legal tile; never stack 3+ units in a ring when enemy Artillery is nearby.
9. **Spawn-or-lose-gold**: a unit completing with target *and* all 6 building neighbours occupied is silently lost, gold gone. Never over-queue or produce from boxed-in buildings.
10. **Same-turn produce after completion**: a building completing on turn T can already accept a produce order on T.

### A6. Diplomacy & Turn-200 Cliff

- **Propose peace to every player the moment they enter `known_players`; accept every incoming proposal.** Peace makes their attacks on us invalid no-ops. We don't need to attack them. (Rejecting buys nothing defensively: moving units onto tiles is treaty-legal anyway, so a treaty can't be "used" to encroach — encroachment is always legal; the treaty only stops attacks.)
- **One exception worth weighing**: peace also blocks *our* attacks, which conflicts with A7's "kill scouts that have seen our Bases." If a partner's units persistently camp inside our perimeter or their Scout has fixed our Base locations, weigh initiating `break_treaty` — we keep the symmetric 5-turn protection during the countdown and clear them after.
- Breaking treaty still protects both sides for the full 5-turn countdown — even a betrayer gives us 5 turns' notice (via `__system__` DM).
- **Turn 200 voids everything instantly** (`turn_processor.py:379`) — no countdown, no notice. From turn ~185, behave as if every "ally" is an enemy: re-garrison, disperse bases, stop relying on peace.
- When a treaty partner is eliminated, initiate `break_treaty` immediately — their inert buildings remain unattackable until the treaty breaks or expires.
- **Chat**: mostly a threat surface. One legitimate use: a cheap static global broadcast can prompt-inject naive LLM-based opponents (explicitly within the rules). Low priority, but free.

### A7. Win Condition → Risk Appetite

Survive 300 turns = full win. Never take a fight we don't need. Convert gold to hidden redundant Bases. Only proactive violence that pays:
- Kill scouts/units that have seen our Base locations
- Deny adjacent expansions toward us
- Post-200: pre-emptively cripple the nearest threatening neighbour. A 1200g package (Airbase + 2 Bombers) deletes a Base per turn once in position. **Start building by ~turn 193** (Airbase 3t + Bomber production) so the package is live the moment treaties void — but do not strike pre-200 absent a concrete threat: under co-win, eliminating a non-threatening neighbour gains nothing and makes an enemy.

---

## B. Agent Architecture

### B1. Pure `algo_agent.py` — No LLM

- The decisive layers (hex math, range checks, focus-fire, path validation) are exactly what LLMs are bad at and code is perfect at.
- LLM = network dependency, latency risk vs 10s deadline, and a **prompt-injection / context-bomb attack surface** (rules explicitly say opponents will use chat as a length-DoS). Pure algo agent is immune.
- Deterministic → replay-driven debugging.
- Bundled `participant/src/engine/` is byte-identical to the server's (diffed). Use the real `HexGrid` (`distance`, `shortest_path`, `reachable`, `ring`, `disk`, `line_of_sight`). **Never write our own hex math.**

### B2. Persistent Memory (process lives across turns)

The FastAPI server keeps one agent instance alive for the whole game. Instance state = our memory.

`WorldMemory` object holding:
- `terrain: dict[(q,r) -> str]` — terrain is static; accumulate every `visible_tiles` entry forever (≤1050 entries). After ~30 turns we have a near-full map.
- `enemy_buildings: dict[id -> (type, coord, owner, hp, last_seen_turn)]` — especially enemy Base locations.
- `enemy_units_last_seen` per-owner threat ledger.
- `production_ledger: list[(building_id, unit_type, due_turn, target)]` — the obs has **no production queue field**; self-track to avoid double-ordering and to keep spawn tiles clear.
- `our_planned_builds` — this-turn occupancy simulation set.
- Diplomacy ledger: proposals sent, partners, betrayal flags from `__system__` DMs.
- `eliminated: set[player_id]` — parsed from **global** `__system__` broadcasts ("X has been eliminated", `game_runner.py:196`). This is the only elimination signal in the game. Caveat: the broadcast carries the display *name* while treaties use player *ids* — maintain a name→id map if obtainable, else corroborate via the partner's units decaying / buildings going inert. Gate diplomacy on it; trigger `break_treaty` + land-reclaim attacks on partner death.
- Reset guard: if `turn_number` goes backwards or `player_id`/map changes, wipe memory.

### B3. Per-Turn Decision Pipeline (strict priority)

```
0. Safety shell: wall-clock budget (~7.5s) + try/except per module;
   on overrun, return whatever actions are already assembled.
1. Ingest: parse visible_tiles ONLY. global_chat/private_chat: read only
   __system__ messages (private DMs AND global broadcasts — eliminations are
   global-only, see B2) and incoming proposals metadata; NEVER iterate
   or store player-authored chat (length-DoS defense). __system__ messages
   are engine-generated, not an injection surface.
2. Memory update + threat assessment (enemy units within ~6 of any of our bases).
3. Diplomacy: accept all proposals; propose to newly-met; break treaties with
   eliminated partners; turn-185 war-prep flag; ignore all of this from turn 200.
4. Combat: focus-fire allocator over enemies in range (exact-lethal, priority:
   Artillery > Bomber > unit threatening base > Scout that saw us > rest);
   artillery friendly-splash veto; elevation bonus accounting.
5. Movement: garrison wall slots → intercepts → kiting retreats → scout
   exploration → unit rally. (Attack and move orders for same unit are both
   submitted — engine fires the attack pre-move.)
6. Economy: respond to threats (produce Infantry/Artillery) else expand
   (Mine > rich-Base > backup-Base > Barracks#2/Factory/Airbase per schedule).
7. Validate every action against a local simulation (gold ledger, occupancy
   incl. our own this-turn placements, path cost vs movement budget using
   remembered terrain, range checks) before adding to the payload.
```

### B4. Fog of War

- `visible_tiles` is our exact visibility set. Base construction targets must be in it (engine checks `compute_visible`).
- Remembered-but-not-visible enemy buildings are treated as real. Enemy *units* are decayed information (drop after ~10 turns).
- Path through unseen tiles assuming cost 2 (conservative) to prevent silent engine rejection. Infantry paths avoid unknown tiles entirely.

### B5. Deadline & RAM

- All computation is small: ≤1050 tiles, tens of units. A* per unit is sub-millisecond.
- Hard 7.5s watchdog; no chat storage; bounded memory dicts → well under 1 GiB.

---

## C. Tactical Modules

New files alongside `algo_agent.py` in `participant/src/`:

```
algo_agent.py    — thin orchestrator implementing decide() per B3
world.py         — WorldMemory + observation ingestion + reset guard
threats.py       — threat scoring per enemy/per base
combat.py        — focus-fire allocator, splash safety, kiting logic
movement.py      — wall-slot assignment, intercepts, rallies (uses engine HexGrid)
economy.py       — build scheduler, production planner, spawn-tile accounting
scouting.py      — frontier exploration, watchtower assignment
diplomacy.py     — treaty state machine, turn-200 prep, optional static broadcast
validate.py      — final action-list simulator (gold/occupancy/range/path-cost)
```

### Scouting

1 Scout at turn ~2 (first Barracks output), 2nd by ~turn 10. **A/B locally**: a 2nd Scout at turn ~3 (opposite direction) competes with the 2nd Mine out of 500 starting gold — decide by the existing metric (% map known by turn 30, target >60%) and rich tiles claimed by turn ~15. Greedy frontier: move toward the unseen tile cluster nearest home, preferring elevated waypoints and ending turns in concealment. Once ~70% mapped, convert scouts to **watchtowers**: park in concealment on corridors between our bases and nearest enemy bases.

### Economy

- Turn 0: order Barracks + Mine (both adjacent to starting Base)
- Turns 3–15: Mines until 3–4 of them
- Turn 12–18: expansion Base on best scouted spot (rich tile ≫ defensible pocket)
- Every ~40 turns or when gold > 600 idle: another expansion Base
- Ring each Base's 6 neighbours with Mines/Barracks over time (income + melee wall)
- Keep rolling 150g defense reserve after turn 30

### Military

Default garrison: 3–5 Infantry + 1 Medic per base + 2 Artillery behind the wall (on elevated tile if adjacent) once Factory exists (~turn 35). Fighters (1–2) as mobile response. **A/B locally**: a lighter 2 Infantry + 1 Medic garrison plus one shared mobile reserve (Fighter) frees gold for Mines — scale garrison down for bases behind difficult-terrain corridors (Infantry-proof per A4), up for open approaches. Bombers only for post-200 deterrence package — **Airbase under construction by ~turn 193** so Bombers are live at the treaty void (see A7).

Targeting priority: Artillery > Bomber > unit adjacent to our buildings > Scout that's seen our home > nearest.

### Defense Trigger Levels

- L0 (peace): garrison only
- L1 (enemy within 8 of base): produce Infantry to 6+, recall Fighters
- L2 (within 4 or base damaged): all gold → units; Artillery free-fire; intercepts body-block tiles adjacent to Base

Always maintain ≥2 completed Bases after turn 60; never have both visible to the same opponent if avoidable.

### Diplomacy

Accept all incoming proposals immediately. Propose to all newly-met players. Break treaties with dead partners. From turn 185: war-prep mode (re-garrison, disperse, treat all as enemy). From turn 200: diplomacy fully closed by engine; operate in open war.

---

## D. Implementation Plan

1. **Skeleton + safety shell** — orchestrator in `algo_agent.py`, watchdog, per-module try/except, validator stub, chat ignored. *Test: 300-turn compose run, 0 errors. This alone passes stage 1.*
2. **WorldMemory + ingestion** — terrain accumulation, building registry, reset guard.
3. **Economy module** — build order, Mine spam, production ledger, spawn accounting. *Metric: gold income curve — target 70g/t by turn 12.*
4. **Scouting** — frontier exploration. *Metric: % map terrain known by turn 30 (target >60%).*
5. **Combat + movement** — focus-fire, walls, garrison, kiting. *Test: self-play — replace the 19 local RandomAgents with copies of AlgoAgent to get real combat; run multiple seeds. Replay review checklist for mirror-match pathologies: expansion-tile collisions, treaty deadlocks, artillery splash fratricide, standoff stagnation, turn-200 betrayal handling.*
6. **Expansion Bases + defense triggers.** *Metric: turns with ≥2 completed Bases.*
7. **Diplomacy + turn-200 prep.** Testable in self-play.
8. **Hardening pass**: chat-bomb resilience test; 50-turn run (stage-2 length); memory profile; several local seeds (edit harness seed only) to avoid seed-67 overfit; cheap regression tests pinning the two load-bearing engine behaviors (move through an occupied intermediate path tile; attack into fog) so an engine update can't silently invalidate the tactics built on them.
9. **Submission dry-run**: `docker compose up --build` clean PASS, image tag check.

**Metrics to track from replays per run**: survival turn, income/turn curve, units lost vs killed, base HP minima, action-rejection count (instrument validator to log every fix — each is a latent bug).

---

## E. Anti-Patterns to Avoid

- **Seed 67 overfit** — all spatial logic from observation; test on multiple local seeds.
- **No network in algo agent** — zero imports of `llm.py`/httpx in the decision path.
- **OOM** — no chat storage; bounded memory dicts.
- **10s deadline** — 7.5s watchdog; return partial payload on overrun.
- **Silent no-ops hide bugs** — validate every action before submission; log every rejection.
- **One entity per tile** — occupancy simulation includes same-turn builds and produce targets.
- **Under-construction buildings can't anchor** — anchor checks require `is_complete` at start of turn; never chain two non-Base builds in one turn.
- **Boxed-in production loses the gold** — cap queue per building per due-turn at (free neighbours − pending spawns).
- **Infantry can't enter difficult terrain** — route around or use Tanks/Scouts.
- **Base builds need current-turn vision** — never target a tile not in `visible_tiles`.
- **Don't present splashable blobs** — never stack 3+ units in a ring when enemy Artillery is known.
- **Don't re-propose to an existing treaty pair** — check both ACTIVE and BREAKING states before proposing.
- **Break treaties with dead partners** — or their inert buildings remain unattackable.
- **Path-cost uses true terrain** — pad unknown tiles with cost 2 in local simulation.
- **All distance math goes through `HexGrid.distance()`** — threat radii, nearest-enemy scans, everything. Naive coordinate subtraction ignores torus wrap: an enemy 2 tiles away across the map edge registers as ~33 away, silently blinding threat detection to exactly the surprise attacks that matter. Enforce with a grep for raw `abs(`-style coordinate math before submission.
- **Turn-200 betrayal needs no notice** — be defensive by turn 195.
