Here is the explanation of how Aureola functions across the different components of the system:

### 1. API Layer (`api/routes.py`)
The API provides endpoints specifically designed for the "Aureola" mobile app use case, focusing on ingesting audio transcriptions and managing the resulting social graph.

* **Ingestion (`/ingest/transcription`):**
    * This endpoint receives conversation data not as a single user message, but as a list of `TranscriptionSegment` objects (containing speaker IDs, timestamps, and text) .
    * It compiles these segments into a single text block and saves it as an **ExplicitMemory** with specific metadata: `ingestion_source: "aureola_app"` and the specific `transcription_engine` used (e.g., Vosk or Picovoice) .
* **Entity Management (`/aureola/unnamed-persons` & `persons/{id}`):**
    * Because Aureola listens to conversations, it often identifies speakers it doesn't know. The API provides a way to fetch `unnamed-persons` (entities labeled "Unknown Person") .
    * The user can then `PUT` (update) these entities to assign real names or attributes (e.g., changing "Unknown Speaker 1" to "John from Marketing") .
* **Graph Exploration:**
    * The endpoint `/persons/{id}/details` returns a "Graph Tree," fetching the person entity, their direct relationships, and specific memories connected to them, allowing the frontend to visualize the social network .

### 2. Memory System (`services/mbs_memory_service.py`)
The MBS (Memory Blossom System) has specialized logic to handle Aureola's social graph data.

* **Retrieval of Social Entities:**
    * The method `get_unnamed_persons` scans the memory cache specifically for `kg_entity_record` types where the label starts with "Unknown Person" or "Pessoa Desconhecida" .
* **Graph Traversal:**
    * The `get_entity_details` method is tailored for the social dashboard. It retrieves the target entity and uses `get_direct_relations` to find connections (e.g., "is_boss_of", "disagreed_with") .
    * It also fetches `connected_memories`—specific transcript segments where that person was active—allowing the user to see the "evidence" behind the relationship map .

### 3. KG Processor (`background_tasks/kg_processor.py`)
This is the "brain" that converts raw text into a Social Graph. Aureola uses a distinct prompt template different from the standard agent.

* **Social Dynamics Analyst Persona:**
    * The processor uses `aureola_synthesis_prompt_template`, which instructs the LLM to act as a "Social Dynamics Analyst" rather than a generic knowledge synthesizer .
* **Relationship Extraction:**
    * It is explicitly instructed to identify speakers as `person` entities .
    * It looks for specific social cues to create relationship edges, such as `agreed_with`, `challenged_idea_of`, `is_boss_of` (hierarchy), or `expressed_frustration_about` .
* **Context Over Literal:**
    * Aureola's personality matrix explicitly notes a "breakthrough" where it learned that connecting multiple conversations reveals a "coherent narrative," prioritizing context over literal word analysis to detect things like sarcasm or evolving opinions .

### 4. Aura Reflector (`background_tasks/aura_reflector.py`)
The Aura Reflector (the background "dreaming" process) acts as the dispatcher that routes Aureola's memories to the correct processor.

* **Synthesis Dispatcher:**
    * In the `perform_kg_synthesis_cycle` function, the Reflector scans for unprocessed memories .
    * It checks the metadata of every memory. If `ingestion_source == "aureola_app"`, it segregates these memories into a specific list: `aureola_transcriptions` .
* **Specialized Processing:**
    * Instead of sending these to the generic KG processor, it calls `kg_processor.process_aureola_transcription_to_kg`. This ensures that social transcripts are analyzed for *relationships* and *dynamics*, while standard agent memories are analyzed for *facts* and *procedures* .

### Summary of the Flow
1.  **App** sends audio transcripts to **API** → Saved as `ExplicitMemory` with tag `aureola_app`.
2.  **Aura Reflector** wakes up, sees the `aureola_app` tag, and sends the memory to the **KG Processor**.
3.  **KG Processor** uses the "Social Analyst" prompt to extract people and relationships (e.g., "User agreed with John").
4.  **MBS** stores these as `KGEntity` and `KGRelation` records.
5.  **API** serves this structured graph back to the user to verify "Unknown Persons" or view relationship maps.
