# JSON FORMAT

{
    "verb": "string - the base form of the verb",
    "applicable_subjects": [
      "subject_types - the entities that can perform the action"
    ],
    "applicable_objects": [
      "object_types - the entities that can receive the action"
    ],
    "goals": [
      "string - intended outcomes or purposes of the action"
    ],
    "mechanisms": [
      "string - processes, methods, or modes of execution"
    ],
    "tools": [
      "string - instruments, faculties, or enablers needed to perform the action"
    ],
    "required_subject_states": {
      "physical": ["string - physical prerequisites for the subject"],
      "emotional": ["string - emotional prerequisites"],
      "mental": ["string - cognitive/intellectual prerequisites"],
      "positional": ["string - spatial/geospatial positioning of the subject relative to the object"]
    },
    "required_object_states": {
      "physical": ["string - physical prerequisites for the object"],
      "emotional": ["string - emotional prerequisites"],
      "mental": ["string - conceptual prerequisites"],
      "positional": ["string - spatial/geospatial positioning of the object relative to the subject"]
    },
    "final_subject_states": {
      "physical": ["string - physical condition after action"],
      "emotional": ["string - emotional condition after action"],
      "mental": ["string - cognitive/intellectual condition after action"],
      "positional": ["string - spatial condition after action"]
    },
    "final_object_states": {
      "physical": ["string - physical condition after action"],
      "emotional": ["string - emotional condition after action"],
      "mental": ["string - conceptual condition after action"],
      "positional": ["string - spatial condition after action"]
    },
}


# Allowed Subject/Object Types (Global Definitions)

  These are the only valid categories for applicable_subjects and applicable_objects:
  
  biological_body → humans, animals, plants, and other organisms
  
  micro_biological_entities → microorganisms, bacteria, fungi, viruses
  
  organization → groups, companies, institutions, governments
  
  physical_agent → machines, robots, hardware devices
  
  synthetic_agent → software-driven entities
  
  autonomous_agent → AI, bots, automated systems
  
  interactive_app → mobile/web/desktop applications
  
  object → tangible physical items, artifacts, or tools
  
  virtual_object → documents, code, digital media, datasets
  
  concept → abstract ideas, theories, mental models
  
  place → spatial/geographical locations, positions, environments
  

# Standardization Notes
  
  Positional Always Geospatial → must reflect spatial relation (e.g., “close to”, “above”, “inside”, “adjacent”), never metaphorical.
  
  Goals vs. Mechanisms
  
  Goals = why the verb is performed (purpose).
  
  Mechanisms = how it is executed (process).
  
  Tools vs. Mechanisms
  
  Tools = external or internal instruments (hands, pens, software).
  
  Mechanisms = the operative method (rotation, speech, analysis).
  
  Subjects vs. Objects
  
  Subjects are actors/initiators of verbs.
  
  Objects are receivers/targets of verbs.  