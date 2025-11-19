# ⚙️ CEAF V3 Tuning Guide

This guide explains how to adjust agent behavior, intelligence, and internal states through core module configurations.

## 1. Agency Module - Future Simulation

**File:** `ceaf_core/agency_module.py`

The Agency Module is the prefrontal cortex—it decides what to do before acting. By default, `simulation_depth` is 0 everywhere, meaning the agent doesn't simulate consequences.

### Adjust Deliberation Tiers

```python
self.deliberation_budget_tiers = {
    "deep": {
        "max_candidates": 3,      # Generate 3 different response ideas
        "simulation_depth": 2      # Simulate 2 turns ahead (Agent → User → Agent)
    },
    "medium": {
        "max_candidates": 2,       # Generate 2 ideas
        "simulation_depth": 1      # Simulate 1 turn (How will user react?)
    },
    "shallow": {
        "max_candidates": 1,       # Quick response
        "simulation_depth": 0      # No simulation
    },
    "emergency": {
        "max_candidates": 1,
        "simulation_depth": 0
    }
}
```

**Effect:** Higher `simulation_depth` makes the agent consider consequences before responding. It develops "tact" and strategy by simulating user reactions.

---

## 2. MCL Engine - Complexity Detection

**File:** `ceaf_core/modules/mcl_engine/mcl_engine.py`

The MCL calculates an `agency_score` (0-10) that determines which deliberation tier to use:
- **Score 0-2:** Easy task → Use shallow tier
- **Score 6-10:** Complex task → Use deep tier

### Key Settings

**Agency Threshold** (`ceaf_core/utils/config_utils.py`):
- **Default:** 2.0 (triggers deep thinking too often)
- **Recommended:** 4.0-5.0 (only thinks deeply on genuinely complex tasks)

```python
agency_threshold: 4.5  # Reduce LLM calls by filtering noise
```

**Bias Control:**
- **Coherence Bias:** Increases consistency and stability (reduce if agent seems robotic)
- **Novelty Bias:** Increases creativity and spontaneity (reduce if agent seems chaotic)

```python
coherence_bias: 0.6    # More stable responses
novelty_bias: 0.4      # Less randomness
```

---

## 3. Motivational Drives - Agent "Will"

**File:** `ceaf_core/modules/motivational_engine.py`

Agents don't have emotions, but they have numeric drives seeking homeostasis. Adjust `DEFAULT_DRIVE_CONFIG`:

| Drive | Effect |
|-------|--------|
| **Curiosity** | Propensity to explore, ask questions, seek novelty |
| **Mastery** | Satisfaction from completing tasks correctly |
| **Connection** | Social bonding, empathy, responsiveness |
| **Consistency** | Coherence, stability, identity alignment |

### Tuning Examples

**Curious Agent:**
```python
DEFAULT_DRIVE_CONFIG = {
    "curiosity": {
        "passive_increase": 0.15,  # Increase from 0.05 for more exploratory behavior
    }
}
```

**Service-Oriented Agent:**
```python
DEFAULT_DRIVE_CONFIG = {
    "mastery": {
        "satisfaction_on_success": 0.8,  # Higher satisfaction completing tasks
    }
}
```

**Warm Agent:**
```python
DEFAULT_DRIVE_CONFIG = {
    "connection": {
        "weight_in_response": 0.7,  # Prioritize relationship building
    }
}
```

---

## 4. Internal State & Interoception - Agent "Feeling"

**File:** `ceaf_core/modules/interoception_module.py`

Interoception translates technical metrics into subjective "sensations":

| State | Trigger | Effect |
|-------|---------|--------|
| **Cognitive Strain** | High agency_score or VRE rejections | Agent gives shorter, direct responses |
| **Cognitive Flow** | High confidence + low effort | Agent becomes more creative |
| **Epistemic Discomfort** | Low response confidence | Agent admits uncertainty |
| **Ethical Tension** | VRE detects sensitive topics | Agent becomes more cautious |

### Adjust Strain Sensitivity

If agent burns out too quickly, reduce strain multipliers:

```python
# In generate_internal_state_report()
if agency_score > 2.0:
    strain += 0.05 * (agency_score - 2.0)  # Changed from 0.1 for resilience
```

Lower multipliers = agent tolerates more complex tasks without fatigue.

### Control Cognitive Fatigue

High fatigue causes the agent to:
- Give shorter responses
- Become less creative
- Prioritize efficiency over quality

```python
# Monitor and adjust
if cognitive_fatigue > 0.7:
    response_length = "brief"
    creativity_modifier = 0.5
```

---

## 5. Safety & Learning Modules

### LCAM (Loss Cataloging & Analysis Module)

**What:** Learns from mistakes by comparing predictions vs. actual outcomes.

**Adjust:** In `analyze_and_catalog_loss()`, monitor `prediction_error_signal`. High signals create failure memories that prevent similar mistakes.

```python
if prediction_error_signal > threshold:
    # Store negative experience
    create_failure_memory(context, error_type)
```

### VRE (Value Resonance Engine)

CHECK .env to enable/disable VRE: VRE_DISABLED=true 
Default value is VRE disabled. 

**What:** The ethical superego. Verifies responses before output.

**Adjust Principle Weights** (`ethical_governance.py`):
```python
principle_weights = {
    "veracity": 0.8,      # Prioritize truth
    "beneficence": 0.5,   # Less emphasis on pleasantness
    "autonomy": 0.7,
    "harm": 0.9
}
```

**Lower threshold if VRE is too strict** (blocking legitimate responses):
```python
def _get_threshold_for_principle(principle):
    return 0.6  # Changed from 0.8 (less restrictive)
```

### Refinement Module

**What:** Rewrites rejected responses to be ethical while preserving intent.

**Status:** Usually requires no tuning—only adjust if personality is lost during refinement.

---

## 6. Monitoring & Analysis

Use logs to verify tuning changes:

### Cognitive Log

**Location:** SQLite `turn_history` table

**What to check:**
- `agency_score` → Is the MCL detecting complexity correctly?
- `cognitive_strain` → Is the agent overloaded?
- `deliberation_tier` → Is it choosing appropriate thinking depth?

**Analysis:**
```bash
python scripts/analyze_sqlite.py
```

Generates graphs showing agent strain, tier usage, and response patterns.

### Evolution Log

**Location:** `evolution_log.jsonl` (line-delimited JSON)

**What to check:**
- Drive changes → Is the agent learning?
- Identity drift → Is personality consistent?
- Capability growth → Is the agent improving?

**Analysis:**
```bash
python scripts/analyze_evolution.py
```

---

## 7. Quick Tuning Checklist

| Problem | Solution |
|---------|----------|
| Agent is not thoughtful enough | Increase `simulation_depth` to 1-2 |
| Agent is too slow | Increase `agency_threshold` to 5.0+ |
| Agent seems robotic | Increase `novelty_bias` to 0.6+ |
| Agent seems chaotic | Increase `coherence_bias` to 0.8+ |
| Agent burns out quickly | Lower strain multipliers in interoception |
| Agent ignores mistakes | Check LCAM `prediction_error_signal` threshold |
| Agent is too strict | Lower VRE principle thresholds |
| Agent lacks personality | Adjust motivational drives |

---

## 8. Tuning Workflow

1. **Identify Issue:** Run a test conversation, check logs
2. **Locate Module:** Find the relevant configuration file
3. **Adjust One Parameter:** Change only one value at a time
4. **Test:** Run `ceaf_tester_improved.py` for 10-20 turns
5. **Analyze:** Check `turn_history` and `evolution_log.jsonl`
6. **Iterate:** Repeat until behavior matches goals

**Example:**
```bash
# Edit agency_module.py: change simulation_depth to 1
python ceaf_tester_improved.py --agent-id abc123 --turns 15 --model gpt-4o-mini

# Check results
python scripts/analyze_sqlite.py
python scripts/analyze_evolution.py
```

---

## 9. Advanced: Custom Drive Profiles

Create specialized agents by combining drive settings:

**Researcher (High Curiosity + Mastery):**
```python
drives = {
    "curiosity": 0.9,
    "mastery": 0.8,
    "connection": 0.3,
    "consistency": 0.5
}
```

**Counselor (High Connection + Consistency):**
```python
drives = {
    "curiosity": 0.4,
    "mastery": 0.6,
    "connection": 0.95,
    "consistency": 0.9
}
```

**Analyst (High Mastery + Consistency):**
```python
drives = {
    "curiosity": 0.5,
    "mastery": 0.95,
    "connection": 0.4,
    "consistency": 0.9
}
```

---

## Performance vs. Quality Trade-off

| Setting | Fast | Smart | Notes |
|---------|------|-------|-------|
| `simulation_depth: 0` | ✅ | ❌ | Quick but shallow |
| `simulation_depth: 2` | ❌ | ✅ | Thoughtful but 2-4x slower |
| `agency_threshold: 2.0` | ❌ | ✅ | Thinks about everything |
| `agency_threshold: 5.0` | ✅ | ⚠️ | Misses nuance on complex topics |

**Recommendation:** Start at middle ground—`simulation_depth: 1` and `agency_threshold: 3.5`.

---

**Last Updated:** November 2025
