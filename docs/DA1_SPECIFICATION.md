# Digital Assignment 1 (DA1) Specification & Literature Survey

**Course:** BCSE306L - Artificial Intelligence  
**Title:** Utility-Driven Coordinated Multi-UAV Exploration with Target Detection and Localization  
**Faculty:** VIJAYAPRABHAKARAN  
**Team Members:** Adarsh H Pillai (24BRS1082) & SaiAmirthesh (24BYB)  
**Repository:** https://github.com/i-sonu/ai-multi-uav-swarm-searcher  

---

## 1. Problem Identification

### 1.1 Application Domain
This project addresses autonomous aerial exploration of unknown environments for post-disaster search and reconnaissance. The task is to have a small team of unmanned aerial vehicles enter an area for which no prior map exists, build a map of it cooperatively, and simultaneously record the location of targets of interest (in the motivating scenario, people) so that limited human rescue effort can be directed to where it is most likely to matter.

Emergency response literature consistently identifies the first 72 hours after a structural collapse as the **"golden period"**: survival probability is highest in the first 24 hours and falls steeply thereafter. Search effort expended in the wrong place during that window is effectively irrecoverable.

Unmanned aerial vehicles (UAVs) can survey large and inaccessible areas rapidly. However, their effectiveness depends on two capabilities that are usually developed in isolation: deciding where to fly next in an unmapped environment, and reliably recognizing small, sparse human figures in aerial imagery. Combining the two (letting what the vehicle detects influence where it searches next, and recording where each detection occurred) is the key gap this project targets.

### 1.2 Stakeholder and Supported Decision
The primary stakeholder is the **incident commander** of a search-and-rescue or disaster-response team, together with the UAV operators reporting to them. The decision the system supports is **resource-directing**: given a bounded but unmapped search area and a limited number of ground rescue teams, which regions have already been searched, which remain unsearched, and where have candidate targets been observed? The system's output converts an unstructured search area into a prioritized, spatially indexed picture of what is known and what has been found.

---

## 2. Literature Survey & Comparative Analysis

The survey covers three converging strands: (i) classical frontier-based exploration and its multi-robot extensions, (ii) recent multi-UAV and learning-based cooperative exploration, and (iii) aerial target detection for search and rescue.

| Ref. | Year | Dataset / Environment | Method / Architecture | Key Metric & Value | Stated Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [1] | 1997 | Simulated + real indoor occupancy grids | Frontier-based exploration; robot drives to nearest boundary between free and unknown cells | Demonstrated full coverage of office environments | Single robot only; greedy nearest-frontier choice is myopic and revisits regions |
| [2] | 1998 | Multi-robot indoor simulation | Extension of frontier exploration to multiple robots sharing a map | Reduced exploration time vs. single robot | No explicit coordination — robots may select the same frontier, causing redundant coverage |
| [3] | 2005 | Indoor office environments, real robot teams | Centralised coordinated exploration; utility-cost trade-off assigns robots to distinct frontiers | Significant reduction in total exploration time vs. uncoordinated baseline | Centralised: heavy communication and computation cost; poor scaling with team size |
| [4] | 2016 | 3D indoor/outdoor simulation | Receding-horizon Next Best-View (NBV) planner using RRT sampling in 3D | Enabled 3D volumetric exploration with onboard compute | Sampling-based NBV is computationally heavy; single-agent formulation |
| [5] | 2020 | 3-D indoor environments | Incrementally built topological map for autonomous exploration | Improved exploration efficiency vs. dense grid methods (IEEE T-IM) | Topological abstraction can miss fine structure; single-agent |
| [6] | 2022 | Disaster-response task scenarios | Multi-robot task allocation with deadlines, range and payload constraints | Improved task completion under deadline constraints | Assumes known task locations — not coupled to online exploration |
| [7] | 2022 | Subterranean / intermittent connectivity settings | ACHORD: communication-aware multi-robot coordination | Maintained coordination under intermittent connectivity (IEEE RA-L) | Substantial infrastructure assumptions; complex comms stack |
| [8] | 2023 | Forest environments, multi-UAV | Fast decentralised multi-UAV exploration | Fast exploration in cluttered forest (IEEE RA-L 8(9)) | Decentralised design assumes reliable local sensing; heavy onboard compute |
| [9] | 2023 | Terrestrial simulation environments | Cooperative autonomous exploration via explicit task allocation | Reduced overlap vs. uncoordinated frontier baseline | Ground robots only; allocation not evaluated under sensing noise |
| [10]| 2024 | Multi-robot simulation benchmarks | Transformer-based reinforcement learning for autonomous exploration | Outperformed classical frontier baselines (Sensors 24(16):5083) | Requires large-scale training; sim-to-real transfer unvalidated |
| [11]| 2024 | Distributed multi-UAV, bandwidth-limited | LECES: low-bandwidth collaborative exploration system | Efficient collaboration at reduced bandwidth (IEEE RA-L 9(9)) | Optimised for comms efficiency rather than exploration optimality |
| [12]| 2024 | Indoor drone imagery | YOLO-IHD: improved real-time human detection for indoor drones | Real-time human detection on drone hardware (Sensors 24:922) | Detection only — no coupling to exploration or mapping |
| [13]| 2025 | VisDrone (pre-train) + Heridal (fine-tune) | YOLOv5s with PB-FPN and deconvolution; two-stage transfer learning | mAP@50 = 0.802 on Heridal, real-time on Jetson Nano | Single-modality RGB; no integration with autonomous flight planning |
| [14]| 2025 | Survey of aerial SAR person-detection benchmarks | Survey and benchmark of aerial person detection for search and rescue | Consolidated benchmark comparison (J. Remote Sensing 5:0474) | Survey — highlights that detection is evaluated offline, decoupled from mission planning |
| [15]| 2025 | Bandwidth-limited multi-UAV simulation | PC-Explorer: decentralised multi-UAV exploration | Exploration under bandwidth limits (IEEE RA-L 10(9)) | Focus on communication constraints; no semantic/target-aware objective |
| [16]| 2025 | Communication-constrained multi-robot settings | Collaborative exploration under communication constraints | Improved coverage under constrained comms (IEEE Access 13) | Geometric coverage objective only — no target-detection reward |
| [17]| 2025 | Multi-robot simulation | Deep RL with knowledge distillation for collaborative exploration | Improved efficiency vs. non-distilled RL (Mathematics 13(1):173) | Training cost high; generalisation to unseen maps limited |
| [18]| 2025 | Systematic review, 2013–2024 corpus | Systematic literature review on multi-robot task allocation | Taxonomy of MRTA methods (ACM Comput. Surv.) | Review notes most MRTA work assumes tasks are known a priori, not discovered online |

### 2.1 Research Gaps Addressed
1. **Exploration and detection are optimized separately:** Existing vision models are evaluated offline on static datasets without coupling to flight policies. This work closes the loop so detections are spatially registered into the map driving exploration decisions.
2. **Coordination benefit is rarely isolated:** The marginal contribution of coordinated vs. uncoordinated allocation while holding team size and planner constant is seldom reported as a clean ablation.
3. **Recent focus concentrates on comms constraints rather than allocation quality:** Small-team allocation optimization remains under-examined.
4. **Learning-based exploration lacks transparent classical baselines:** Reinforcement learning approaches carry high training costs and under-specified baselines.
5. **Task allocation literature assumes tasks are known in advance:** In exploration, frontiers are non-stationary and generated dynamically online.

---

## 3. Detailed Problem Statement

Given synchronous 2-D LiDAR range scans and downward-facing RGB imagery from two simulated UAVs deployed into a bounded **60 m × 60 m environment** discretised into a shared **240 × 240 occupancy grid at 0.25 m resolution**, produce:
- **(a)** a complete free/occupied/unknown labelling of the environment, and
- **(b)** a georeferenced register of detected targets with grid coordinates, class label, confidence, and first-detection provenance.

**Success Criteria:**
- **Time-to-90%-coverage:** Target at least a **35% reduction** over the single-agent nearest-frontier baseline B1.
- **Redundant Coverage Ratio:** Measurably lower than the two-agent uncoordinated baseline B2.
- **Target Detection Quality:** Reported as mAP@50 on a held-out, scale-stratified test split.

---

## 4. References

1. B. Yamauchi, "A frontier-based approach for autonomous exploration," in *Proc. IEEE CIRA*, 1997.
2. B. Yamauchi, "Frontier-based exploration using multiple robots," in *Proc. 2nd Int. Conf. Autonomous Agents*, 1998.
3. W. Burgard et al., "Coordinated multi-robot exploration," *IEEE Trans. Robot.*, vol. 21, no. 3, 2005.
4. A. Bircher et al., "Receding horizon 'next-best-view' planner for 3D exploration," in *Proc. IEEE ICRA*, 2016.
5. C. Wang et al., "Efficient autonomous exploration with incrementally built topological map in 3-D environments," *IEEE Trans. Instrum. Meas.*, 2020.
6. P. Ghassemi and S. Chowdhury, "Multi-robot task allocation in disaster response," *Robot. Auton. Syst.*, 2022.
7. M. Saboia et al., "ACHORD: Communication-aware multi-robot coordination," *IEEE RA-L*, 2022.
8. L. Bartolomei et al., "Fast multi-UAV decentralized exploration of forests," *IEEE RA-L*, 2023.
9. "Multi-robot cooperative autonomous exploration via task allocation in terrestrial environments," *Front. Neurorobot.*, 2023.
10. Q. Chen et al., "Transformer-based reinforcement learning for multi-robot autonomous exploration," *Sensors*, 2024.
11. T. Zhang et al., "LECES: A low-bandwidth and efficient collaborative exploration system," *IEEE RA-L*, 2024.
12. G. Kucukayan and H. Karacan, "YOLO-IHD: Improved real-time human detection system for indoor drones," *Sensors*, 2024.
13. "Real-time search and rescue with drones: A deep learning approach for small-object detection based on YOLO," *Drones*, 2025.
14. X. Zhang et al., "Aerial person detection for search and rescue: Survey and benchmarks," *J. Remote Sens.*, 2025.
15. Y. Hui et al., "PC-Explorer: Decentralized multi-UAV exploration in bandwidth-limited environments," *IEEE RA-L*, 2025.
16. G. Lu et al., "Multi-robot collaborative exploration on communication-constrained environments," *IEEE Access*, 2025.
17. R. Wang et al., "A multi-robot collaborative exploration method based on deep reinforcement learning and knowledge distillation," *Mathematics*, 2025.
18. "A systematic literature review on multi-robot task allocation," *ACM Comput. Surv.*, 2025.
19. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed., Prentice Hall, 2015.
20. CRED, *2024 Disasters in Numbers (EM-DAT Annual Report)*, 2025.
21. G. E. M. Abro et al., "Synergistic UAV motion: A comprehensive review on advancing multi-agent coordination," *ICCK Trans.*, 2024.
22. L. Yang et al., "Multi-UAV collaborative target search method in unknown dynamic environment," *Sensors*, 2024.
EOF"
