AI-Driven Cloud Resource Manager



Development Specification

This document is the development source of truth for the project.

Changes to module boundaries, architecture, or major technologies must be discussed by all team members before implementation.

1\. Project Purpose

The project is a cloud-hosted OS resource-manager simulator that demonstrates how AI/ML can assist operating-system resource management.

The system focuses on three OS functionalities:

CPU / Process Management
Memory Management
Device Management
The core concept is:
PREDICT → DECIDE → ACT → MEASURE



AI is used to make predictions or decisions, while classical OS algorithms and resource-management mechanisms perform the actual actions.



2\. Frozen Module Structure



The project contains exactly THREE core modules.



Module 1 — AI-Assisted CPU Scheduling



OS Functionality



CPU / Process Management



Classical Algorithms



FCFS



SJF



Priority Scheduling



Round Robin



AI Components



CPU burst-time prediction using regression



Workload/job-mix classification



AI-assisted meta-scheduler



Core Flow



Process/workload input

→ burst-time prediction

→ workload classification

→ scheduler selection

→ classical scheduling algorithm

→ execution

→ metrics



Outputs / Metrics



Gantt chart



Waiting time



Turnaround time



Response time



Completion time



CPU utilization



Scheduling overhead



Design Principle



AI assists in selecting the scheduling strategy.



The selected classical scheduling algorithm performs the actual scheduling.



3\. Module 2 — Predictive Memory Management



OS Functionality



Memory Management



Module 2 remains ONE module even though it contains multiple memory-management components.



3.1 Page Replacement



Classical Algorithms



FIFO



LRU



Optimal



AI / Prediction



Markov next-page predictor



Transformer-based next-page predictor



AI Action



The predictor may:



prefetch a predicted page



protect a predicted page from replacement



Metrics



Page faults



Hit ratio



Miss ratio



Execution time



Prediction accuracy



3.2 Dynamic Memory Allocation



Classical allocation algorithms:



First Fit



Best Fit



Worst Fit



3.3 System Memory Monitoring



Use psutil for:



RAM monitoring



Swap monitoring



Per-process memory logging



AI-assisted components include:



memory-pressure forecasting



abnormal memory-growth / leak detection



Possible actions:



generate an alert



perform a controlled restart of an offending process



4\. Module 3 — Device Management \& Predictive Maintenance



OS Functionality



Device Management



The module contains both simulated device management and disk predictive maintenance.



4.1 Device Simulation



Simulated devices may include:



Disk



Printer



Network Interface Card (NIC)



Sensor



Telemetry can include:



Temperature



Error rate



Response latency



Queue length



I/O activity



4.2 Device Anomaly Detection



Use a Transformer-based sequence anomaly detector.



The detector is primarily trained using normal telemetry.



The system calculates an anomaly score and applies a threshold to determine whether a device is unhealthy or degrading.



Action



When a device is identified as unhealthy:



stop assigning new I/O to the device



reroute I/O to a backup device



Metrics



Anomaly detection quality



False positives



Detection lead time



Precision



Recall



Reroute time



4.3 Disk Predictive Maintenance



Use:



SMART attributes



Backblaze disk data



Possible models:



Logistic Regression



Random Forest



The model predicts disk failure risk.



Action



When the failure-risk threshold is reached:



trigger cloud backup



Logging



Record:



prediction



risk



action



backup result



Metrics



Precision



Recall



Failure-risk prediction performance



Backup success



5\. Supporting Components



The following are supporting components and are NOT separate project modules:



Data collection



Data generation



ML layer



Backend



Dashboard



Cloud deployment



Storage



Logging



Testing infrastructure



Data collection/generation is not a module.



These components support the three core OS modules.



6\. High-Level Architecture



The development architecture follows:



Input / Data Sources

&#x20; ↓

Backend / OS Resource Manager

&#x20; ↓

&#x20;┌───────────────┬──────────────────┬────────────────────────┐

&#x20;│ │ │

&#x20;CPU Module Memory Module Device Module

&#x20;│ │ │

&#x20;└───────────────┴──────────────────┴────────────────────────┘

&#x20; ↓

ML / Intelligence Layer

&#x20; ↓

Decision / Action Layer

&#x20; ↓

Monitoring / Dashboard



Cloud deployment hosts the integrated system.



Cloud storage is used when disk backup is triggered.



7\. Input / Data Sources



The system can use:



CPU



Synthetic process/workload generator



Memory



Page-reference generator



psutil live memory logs



Device



Simulated device telemetry



SMART disk data



Backblaze disk dataset



8\. ML / Intelligence Layer



The planned AI components are:



CPU



Burst-time regression model



Workload classifier



Meta-scheduler



Memory



Markov next-page predictor



Transformer next-page predictor



Memory-pressure forecaster



Memory leak / abnormal growth detector



Device



Transformer device anomaly detector



SMART disk failure-risk classifier



9\. Decision / Action Layer



Predictions must lead to meaningful system actions.



Examples:



CPU



Select an appropriate scheduling algorithm.



Memory



Prefetch or protect a predicted page.



Memory Health



Generate an alert or perform a controlled process restart.



Device



Flag an unhealthy device and reroute I/O.



Disk



Trigger cloud backup when failure risk crosses the defined threshold.



10\. Monitoring and Outputs



The system should expose:



Predictions



Decisions



Actions



Metrics



Alerts



Logs



Visualizations



The dashboard should allow the integrated system behavior to be demonstrated.



11\. Technology Stack



The initial planned stack is:



Programming



Python



Backend



Flask or FastAPI



Classical OS Algorithms



Plain Python



Machine Learning



scikit-learn



Deep Learning



PyTorch or TensorFlow/Keras



System Monitoring



psutil



Data Processing



pandas



Visualization



matplotlib



Dashboard



Streamlit OR HTML/JavaScript + Chart.js



Cloud



One cloud platform:



AWS EC2



GCP



Azure



Cloud Storage



Possible options:



Amazon S3



Firebase Storage



The final choices should be frozen during setup before unnecessary dependencies are added.



12\. Repository Structure



AI-Cloud-Resource-Manager/

│

├── README.md

├── DEVELOPMENT.md

├── PROJECT\_STATUS.md

├── DECISIONS.md

├── requirements.txt

├── .gitignore

│

├── modules/

│ ├── cpu/

│ ├── memory/

│ └── device/

│

├── ml/

│ ├── cpu/

│ ├── memory/

│ └── device/

│

├── data/

│ ├── cpu/

│ ├── memory/

│ └── device/

│

├── common/

│ ├── config/

│ ├── logging/

│ └── utils/

│

├── dashboard/

│

├── deployment/

│

├── tests/

│ ├── cpu/

│ ├── memory/

│ └── device/

│

└── docs/

&#x20; ├── architecture/

&#x20; ├── results/

&#x20; └── decisions/



13\. Development Roadmap



Development is divided into phases and checkpoints.



Phase 0 — Scope \& Development Setup



CP0.1



Freeze project scope.



CP0.2



Freeze architecture specification.



CP0.3



Set up repository and development environment.



Phase 1 — Classical OS Baselines



CP1.1



CPU scheduler core.



Implement and test:



FCFS



SJF



Priority



Round Robin



CP1.2



Memory page replacement.



Implement and test:



FIFO



LRU



Optimal



CP1.3



Memory allocation.



Implement:



First Fit



Best Fit



Worst Fit



CP1.4



Device simulator.



Implement simulated devices and basic device operations.



CP1.5



Baseline integration sanity check.



Phase 2 — Data Generation \& Data Pipelines



CP2.1



CPU workload/data generator.



CP2.2



Page-reference dataset generation.



CP2.3



Live memory logging using psutil.



CP2.4



Device telemetry dataset generation.



CP2.5



SMART disk data preparation.



Phase 3 — AI Components



CP3.1



CPU burst-time predictor.



CP3.2



CPU workload classifier and meta-scheduler.



CP3.3



Markov page predictor.



CP3.4



Transformer page predictor.



CP3.5



Memory-pressure forecaster.



CP3.6



Memory leak / abnormal memory-growth detector.



CP3.7



Transformer device anomaly detector.



CP3.8



Disk failure-risk classifier.



Phase 4 — Prediction-to-Action Integration



CP4.1



CPU AI action loop.



CP4.2



Memory page prediction action.



CP4.3



Memory health action.



CP4.4



Device recovery action.



CP4.5



Disk backup action.



Phase 5 — System Integration



CP5.1



Common backend interface.



CP5.2



Shared monitoring.



CP5.3



Integrated resource-manager execution.



Phase 6 — Dashboard \& Cloud Deployment



CP6.1



Dashboard skeleton.



CP6.2



Live system outputs.



CP6.3



Cloud VM deployment.



CP6.4



Cloud backup integration.



Phase 7 — Evaluation \& Stress Testing



CP7.1



CPU evaluation.



CP7.2



Memory evaluation.



CP7.3



Device evaluation.



CP7.4



Ablation and fallback checks.



CP7.5



Stress testing.



Phase 8 — Stabilization \& Final Build



CP8.1



Code stabilization.



CP8.2



Reproducibility.



CP8.3



Final development freeze.



14\. Team Ownership



There are three team members.



Member 1



Primary responsibility:



Module 1 — CPU Scheduling



Backend integration lead



Member 2



Primary responsibility:



Module 2 — Memory Management



ML/data pipeline lead



Member 3



Primary responsibility:



Module 3 — Device Management



Dashboard/cloud lead



Important



Ownership means primary implementation responsibility.



It does NOT mean only one person needs to understand that module.



All three members must understand:



architecture



module interactions



AI decisions



evaluation



final integrated system



15\. Git Collaboration Rules



The main branch represents the stable project.



Do not use main as a personal development branch.



Basic workflow:



Create / update feature branch

&#x20; ↓

Implement

&#x20; ↓

Test

&#x20; ↓

Commit

&#x20; ↓

Push branch

&#x20; ↓

Create Pull Request

&#x20; ↓

Another member reviews

&#x20; ↓

Merge into main



Example branches:



main



feature/cpu-scheduling

feature/memory-management

feature/device-management



Later branches should be created for specific checkpoints where useful.



16\. Commit Rules



Use clear commit messages.



Examples:



Add FCFS scheduler



Implement LRU page replacement



Add device simulator



Add CPU burst predictor



Add memory pressure forecasting



Integrate device anomaly detection



Add dashboard skeleton



Avoid messages such as:



update



changes



final



stuff



test



17\. Pull Request Rules



Every meaningful feature should be submitted through a Pull Request.



A Pull Request should contain:



What changed?



Brief description.



Why?



Purpose of the change.



Testing



Explain how the feature was tested.



Related checkpoint



Example:



CP1.1 — CPU Scheduler Core



At least one other team member should review the change before merging it into main.



18\. Checkpoint Rules



Development follows:



PHASE

&#x20; ↓

CHECKPOINT

&#x20; ↓

SMALL STEPS

&#x20; ↓

IMPLEMENT

&#x20; ↓

TEST

&#x20; ↓

VERIFY

&#x20; ↓

CHECKPOINT COMPLETE



Do not skip checkpoints.



Do not begin a later checkpoint if an earlier checkpoint contains unresolved blocking issues unless the team explicitly decides to proceed.



19\. Definition of Done



A checkpoint is considered complete only when:



implementation exists



code runs



basic tests pass



expected output is verified



code is committed



branch is pushed



Pull Request is reviewed where applicable



project status is updated



For AI checkpoints, additionally verify:



dataset/input is documented



preprocessing is documented



model training works



evaluation metrics are recorded



prediction output is saved or accessible



fallback behavior is defined where required



20\. Architecture Change Rule



Do not independently introduce:



a new core module



a new major technology



a new cloud service



a new AI component



a major change to an existing module boundary



Discuss the change with all three members first.



Update:



DEVELOPMENT.md



DECISIONS.md



when a major architectural decision is officially changed.



21\. Development Philosophy



The project should demonstrate a complete resource-management loop rather than isolated ML models.



For every AI component, ask:



What is being predicted?

&#x20; ↓

Why is it being predicted?

&#x20; ↓

What decision uses the prediction?

&#x20; ↓

What action is taken?

&#x20; ↓

How is the action measured?



The goal is an integrated AI-assisted OS resource manager, not simply a collection of machine-learning models.



22\. Current Development State



Current phase:



Phase 0 — Scope \& Development Setup



Current checkpoint:



CP0.1 — Freeze Scope



Completed setup:



GitHub repository created



Local repository cloned



Project directory structure created



.gitignore added



requirements.txt created



Initial structure committed and pushed



Next checkpoint:



CP0.1 — Freeze Scope



No implementation should begin until the team has confirmed the frozen project scope and module boundaries.



23\. Reference Materials



The project development should use the approved project specification and research references stored in the ChatGPT Project Sources / project documentation.



Important references include research concerning:



AI-assisted CPU/cloud scheduling



learning-based page replacement



predictive memory management



device anomaly detection



predictive disk maintenance



cloud resource management



Research papers should be used to support implementation decisions and evaluation design.



24\. Development Control



ChatGPT is used as a development assistant/controller.



When requesting development help, provide:



current phase



current checkpoint



completed work



current problem



expected output



Development should proceed in small steps.



A development request should follow:



PROJECT

→ PHASE

→ CHECKPOINT

→ STEP

→ TASK

→ OUTPUT

→ VERIFY



The assistant should not unnecessarily jump ahead to later checkpoints.



25\. Development Prompt Template



Use this template when opening a new development chat:



PROJECT:

AI-Driven Cloud Resource Manager



PURPOSE:

Cloud-hosted OS resource-manager simulator using AI-assisted CPU scheduling,

predictive memory management, and device management/predictive maintenance.



ARCHITECTURE:

Three core modules:

1\. AI-Assisted CPU Scheduling

2\. Predictive Memory Management

3\. Device Management \& Predictive Maintenance



CURRENT PHASE:

\[Phase]



CURRENT CHECKPOINT:

\[Checkpoint]



COMPLETED SO FAR:

\[List completed work]



CURRENT TASK:

\[Exact task]



CONSTRAINTS:

\- Follow DEVELOPMENT.md

\- Do not change the frozen module structure without team approval

\- Do not create data collection as a separate module

\- Work only on the requested checkpoint

\- Provide small, testable steps



EXPECTED OUTPUT:

\[Expected code/files/result]



TECH STACK:

Python, Flask/FastAPI, scikit-learn, PyTorch or TensorFlow/Keras,

psutil, pandas, matplotlib, Streamlit or HTML/JS + Chart.js,

cloud VM and cloud storage.



INSTRUCTION:

Guide development step by step.

Stop after the requested step/checkpoint and wait for confirmation before proceeding.



26\. Final Principle



The project must remain centered around:



AI-assisted OS resource management



rather than becoming:



a collection of unrelated AI models.



The three core OS modules are the foundation.



AI, data, backend, dashboard, cloud deployment, storage, logging,

and testing exist to support those modules.


