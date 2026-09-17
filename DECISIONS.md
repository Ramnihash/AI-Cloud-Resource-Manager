Project Decision Log



This file records major project decisions agreed upon by the team.



Changes to these decisions should be discussed and agreed upon by all three team members before implementation.



D001 — Core Module Structure



Decision:

The project will contain exactly three core OS modules:



AI-Assisted CPU Scheduling



Predictive Memory Management



Device Management \& Predictive Maintenance



Status: Approved



Reason:

The project is organized around three primary OS functionalities:

CPU/Process Management, Memory Management, and Device Management.



D002 — Data Collection Is Not a Module



Decision:

Data collection and data generation will be treated as supporting components, not as a separate project module.



Status: Approved



Reason:

Data pipelines support the three core OS modules and do not represent an independent OS functionality.



D003 — AI + Classical OS Algorithms



Decision:

AI components will assist prediction and decision-making, while classical OS algorithms will perform the corresponding resource-management operations wherever applicable.



Status: Approved



Examples:



AI predicts CPU burst time / workload → classical scheduler executes scheduling.



AI predicts page access → memory manager performs prefetch/protection action.



AI detects device degradation → device manager reroutes I/O.



AI predicts disk failure risk → backup action is triggered.



D004 — Module 2 Remains One Module



Decision:

Page replacement, dynamic memory allocation, and system memory monitoring will remain components of the single Predictive Memory Management module.



Status: Approved



Reason:

These components all belong to OS memory management and should not be presented as separate core modules.



D005 — Device Management Includes Predictive Maintenance



Decision:

Device simulation, device anomaly detection, and disk predictive maintenance will remain under the single Device Management \& Predictive Maintenance module.



Status: Approved



Reason:

Predictive maintenance is treated as part of device management rather than as an independent module.



D006 — Repository Structure



Decision:

The project will use a shared GitHub repository with separate development branches for team members/features.



Status: Approved



Repository:



AI-Cloud-Resource-Manager



Stable branch:



main



Development work should normally occur on feature branches and be merged through Pull Requests.



D007 — Main Branch Workflow



Decision:

The main branch represents the stable version of the project.



Status: Approved



Workflow:



Feature Branch

&#x20;     ↓

Implementation

&#x20;     ↓

Testing

&#x20;     ↓

Commit

&#x20;     ↓

Push

&#x20;     ↓

Pull Request

&#x20;     ↓

Team Review

&#x20;     ↓

Merge into main



Team members should avoid directly developing experimental features on main.



D008 — Development Specification



Decision:

DEVELOPMENT.md is the shared development specification for the project.



Status: Approved



It defines:



project scope



module boundaries



architecture



technology direction



development phases



checkpoints



team responsibilities



development rules



Major changes should be reflected in DEVELOPMENT.md.



D009 — Architecture Changes



Decision:

Major architecture changes require discussion among all three team members before implementation.



Status: Approved



This includes changes such as:



adding a core module



removing a core module



changing module boundaries



introducing a major technology



introducing a major AI component



changing the cloud architecture



D010 — Development Approach



Decision:

Development will proceed phase-wise and checkpoint-wise.



Status: Approved



The team will:



Complete the current checkpoint.



Test the implementation.



Verify the expected output.



Commit and push the work.



Review/merge where applicable.



Move to the next checkpoint.



The project should not unnecessarily skip ahead to later development stages.



Decision Status Legend



Proposed — Under discussion



Approved — Agreed upon by the team



Changed — Replaced by a newer decision



Rejected — Considered but not adopted

