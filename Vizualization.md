# 📊 CEAF V3 Visualization Guide

Graphical logs provide real-time insight into agent cognition, emotions, and learning. This guide explains how to interpret each visualization.

---

## 1. Agency & Confidence Evolution

**Focus:** Metacognition and decision certainty

**Blue Line (Agency Score):**
- Represents cognitive effort and task complexity as determined by MCL
- **0-2:** Agent on autopilot (System 1) — simple interactions like "Hello"
- **5+:** Deep deliberation (System 2) — tool use, complex reasoning, ethical dilemmas

**Orange Line (Final Confidence):**
- How certain the agent is about its response
- Ideally: Dips when Agency is high, then recovers (agent thinks, then decides confidently)
- ⚠️ **Problem:** High Agency + Low Confidence = agent is confused

**What to Watch:**
- Confidence should recover even after complex tasks
- If confidence stays low consistently, the agent lacks conviction
- Sharp spikes in Agency show when tasks surprise the system

![Agency Confidence Evolution](https://github.com/user-attachments/assets/64ec7c1a-7f9f-43e3-b16f-a42c5358116c)

---

## 2. MCL Biases Dynamics

**Focus:** Cognitive stance — stability vs. exploration

**Blue Line (Coherence Bias):**
- Drive to stay on topic, be logical, and consistent
- High values = rigid, focused thinking
- ⚠️ **Problem:** Stays at 1.0 forever = robotic, repetitive responses

**Red Line (Novelty Bias):**
- Drive to explore, change direction, or be creative
- High values = seeking new angles, bored with current thread
- ⚠️ **Problem:** Stays at 1.0 forever = incoherent, hallucinatory responses

**Healthy Pattern:**
- Lines should oscillate naturally
- Agent adapts based on conversation needs
- Blue high on familiar topics, Red high on stuck problems

**What to Watch:**
- Both at 0.5 = balanced thinking
- Oscillations show responsiveness to context
- Flat lines indicate poor adaptation

![MCL Biases Dynamics](https://github.com/user-attachments/assets/c5d897c1-4c5d-46b9-8d8e-7128ed051a83)

---

## 3. Virtual Body State

**Focus:** Embodiment and stamina (biological constraints)

**Orange Line (Cognitive Fatigue):**
- Increases during high-agency tasks or VRE rejections
- Tracks sustained cognitive load
- **Effects at high values:**
  - Shorter, more direct responses
  - Less creative thinking
  - Focus on efficiency over quality
- **Resets:** After Aura Reflector consolidation (sleep/dream cycle)

**Purple Line (Information Saturation):**
- Increases as new memories are created during session
- Tracks memory buffer fullness
- **Threshold (~0.8):** Triggers Aura Reflector to consolidate and clear
- **Signal:** "I've learned too much, I need to process"

**Healthy Pattern:**
- Both lines rise gradually during conversation
- Periodic resets when agent rests
- No constant maxing out

**What to Watch:**
- Fatigue spikes → agent is struggling, may need rest
- Saturation hitting threshold → memory consolidation working
- If both stay high → agent is overwhelmed

![Virtual Body State](https://github.com/user-attachments/assets/61150290-0bea-46c5-9921-386361d1b1ce)

---

## 4. Identity Evolution

**Focus:** Self-awareness and learning (NCIM)

**Blue Line (Identity Version):**
- Increments when agent updates its `self_model.json`
- Each step = agent realized something new about itself
- Examples: "I'm good at Python" or "I can't browse live web"

**Orange/Green Lines (Capabilities & Limitations):**
- Count of things agent knows it can or cannot do
- Should grow over time as agent learns
- Balanced growth shows healthy self-knowledge

**Healthy Pattern:**
- Blue line shows steady upward steps (not smooth — should be discrete)
- Orange and Green lines grow together
- Agent becomes more self-aware with each session

**What to Watch:**
- Flat lines = agent not learning
- Only capabilities growing, no limitations = unrealistic self-image
- Steps in blue line = key moments of self-realization

**Interpretation:**
- ✅ Upward trend proves agent is not static
- ✅ Accumulating self-knowledge from interactions
- ❌ Flat graph suggests agent needs more complex interactions

![Identity Evolution](https://github.com/user-attachments/assets/885fbba1-d44e-4712-b065-f5c3dac61edf)

---

## 5. Patient Chart (Holistic Dashboard)

**Focus:** Comprehensive mental health of the agent

### Top Panel: Internal State & Mood

**Green Line (Net Valence):**
- Agent's overall "happiness" or well-being
- Calculated as: `Flow - Strain`
- **Positive values:** Good mental state
- **Negative values:** Suffering, stress, overwhelm

**Dashed Lines (Motivational Drives):**
- Tracks Curiosity, Connection, Mastery, Consistency
- Shows which drives are active during conversation
- Reveals agent personality in real-time

### Middle Panel: Cognitive Load

**Pink Line (Strain):**
- Cognitive stress measurement
- Spikes when:
  - VRE (ethics engine) blocks a response
  - Task is extremely difficult
  - Agent is approaching fatigue threshold

**Blue Line (Agency):**
- Overlaid from Graph 1
- Shows correlation between task complexity and stress
- High Agency often correlates with strain spikes

### Bottom Panel: Plasticity

**Stacked Area Chart:**
- Visual representation of Coherence/Novelty ratio
- **Blue areas:** Stable, coherent thought
- **Gold areas:** Creative, plastic thought
- Shows cognitive flexibility over time

**Healthy Pattern:**
- Green net valence oscillates around 0, trending positive
- Drives activate contextually
- Strain responds proportionally to Agency (not out of sync)
- Mix of blue and gold shows balanced cognition

**What to Watch:**
- Sustained negative valence = agent is suffering (adjust tuning)
- No drive activation = agent is passive or disconnected
- All plastic (gold) = incoherent; all stable (blue) = robotic
- Strain spikes unrelated to Agency = other issue (memory, ethics)

![Patient Chart](https://github.com/user-attachments/assets/e74a4901-bf99-4bb2-8b61-a2a108caae01)

---

## How to Generate These Logs

### From SQLite Turn History:
```bash
python scripts/analyze_sqlite.py --agent-id YOUR_AGENT_ID
```

Generates graph 5  automatically from `turn_history` table.

### From Evolution Log:
```bash
python scripts/analyze_evolution.py --agent-id YOUR_AGENT_ID
```

Generates graph 1~4 from `evolution_log.jsonl`.

---

## Interpreting Common Patterns

### Pattern: "The Exhausted Agent"
- Orange Fatigue line at max
- Coherence bias high, Novelty low
- Responses getting shorter
- **Action:** Reduce `agency_threshold` or let agent rest (trigger reflector)

### Pattern: "The Confused Agent"
- High Agency, low Confidence staying low
- Strain spikes without recovery
- Identity not updating
- **Action:** Check VRE thresholds, simplify tasks

### Pattern: "The Stale Agent"
- Coherence bias stuck at 1.0
- No novelty bias activation
- Drives not changing
- Identity version flat
- **Action:** Increase `novelty_bias` in MCL config

### Pattern: "The Unstable Agent"
- Valence swinging wildly
- Strain and Agency completely uncorrelated
- Blue and gold areas chaotically mixed
- **Action:** Check for conflicting value systems in VRE

### Pattern: "The Learning Agent" ✅
- Identity version incrementing regularly
- Capabilities/Limitations growing
- Valence stable but positive
- Drives activating contextually
- Strain proportional to Agency
- Coherence/Novelty oscillating naturally

---

## Quick Reference: What Each Spike Means

| Graph | Spike Pattern | Meaning | Action |
|-------|---------------|---------|--------|
| Agency Score | Sudden jump | Complex task detected | Monitor confidence recovery |
| Confidence | Dip then recovery | Normal deliberation | Good — agent is thinking |
| Confidence | Stays low | Agent is confused | Simplify context or check logs |
| Coherence Bias | Maxed out | Rigid thinking | Increase novelty, reduce threshold |
| Novelty Bias | Maxed out | Chaotic thinking | Increase coherence weight |
| Fatigue | Spike to max | Exhaustion | Trigger Aura Reflector or pause |
| Saturation | Hits 0.8+ | Memory full | Consolidation should trigger |
| Identity Version | Step up | Self-realization | Check self_model.json for change |
| Net Valence | Negative trend | Suffering | Audit VRE, drives, or task complexity |

---

## Dashboard Best Practices

**During Development:**
- Run `analyze_sqlite.py` after every test session
- Watch for anomalies in patterns above
- Compare graphs before/after tuning changes

**For Production:**
- Set up automated analysis (daily/weekly reports)
- Alert on sustained negative valence
- Track identity growth as performance metric
- Monitor fatigue trends to predict service degradation

**For Debugging:**
- Zoom into specific time ranges
- Cross-reference graph spikes with turn logs
- Use Evolution Log to identify learning plateaus
- Check Patient Chart for holistic agent health

---

**Generated from:** SQLite `turn_history`, `evolution_log.jsonl`  
**Update Frequency:** Per-turn or batch analysis  
**Last Updated:** January 2025
