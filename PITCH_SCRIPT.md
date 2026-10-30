**SMART SAHARA — 5-MINUTE PITCH SCRIPT (SINGLE-PAGE A4)**  
**Speaker:** Amine Taghi (South Devs) | **Timing:** 5:00 min (300s)

**[0:00–0:25] SLIDE 1 & 2: INTRO & TEAM**  
Honorable members of the jury, good morning. I am Amine Taghi from South Devs. Today I present **Smart Sahara**—a sovereign, frugal Edge AI solution designed for Laâyoune. Our mission: transform our existing city cameras network from passive video recorders into an autonomous, life-saving public safety engine.

**[0:25–0:35] SLIDE 3: AGENDA**  
I will briefly present the urban challenge, our Edge AI solution, our live working prototype, and the sustainable impact for Morocco and Africa.

**[0:35–1:20] SLIDE 4: CONTEXT & PROBLEM**  
Today, municipal cameras rely on passive DVR systems—they only record footage for humans to review *after* a tragedy. When a citizen collapses from a sudden heart attack, critical minutes are lost. When a child goes missing in crowded markets, authorities lose hours manually scrubbing video feeds. Aligned with Morocco Digital 2030, we asked: Can existing infrastructure actively protect citizens without spending millions on new hardware? Yes.

**[1:20–2:05] SLIDE 5: OUR SOLUTION**  
Smart Sahara delivers three key capabilities:  
1. **Instant Missing Person Search:** Officers type or speak a plain description into our NLP search (e.g., *"child with red cap and yellow shirt"*), and the system tracks that person across cameras.  
2. **Autonomous Emergency Triage:** Instantly flags sudden falls and cardiac collapses with zero human latency.  
3. **100% Privacy-Preserving:** No facial biometrics stored. We track via clothing and color signatures, with real-time face blurring compliant with CNDP Law 09-08.

**[2:05–2:40] SLIDE 6 & 7: ARCHITECTURE & SECURITY**  
Our system runs on four layers:  
• **Perception:** Live feeds from the city cameras network.  
• **Network:** Local municipal fiber streaming via low-latency RTSP/MJPEG.  
• **Platform:** Edge AI running YOLOv8 tracking and NLP search at 30+ FPS on local hardware.  
• **Application:** Command control dashboard with live alerts and GPS dispatch.  
Everything runs on-premises at the edge—guaranteeing 100% national data sovereignty and zero cloud leaks.

**[2:40–3:55] SLIDE 8: LIVE PROTOTYPE DEMO (CORE FOCUS)**  
*(Point to the screen)* Here is our live prototype running on Laâyoune sectors:  
On the top feed at Place du Méchouar, an elderly citizen collapses. The AI verifies the gravitational head-drop and immobility within two seconds, immediately raising an emergency alert.  
Below in Sector A1, we simulate a lost child search. The operator inputs our NLP description; within frames, our part-based Re-ID locks onto the target by clothing colors and pinpoints the exact camera location. Hours of searching become seconds of automated response.

**[3:55–4:25] SLIDE 9: EXPECTED IMPACT**  
• **Social:** 260,000 citizens protected; cuts response time by 80% to rescue people within the golden hour.  
• **Economic:** **0 MAD in new camera Capex.** We upgrade existing city cameras purely through software.  
• **Environmental:** Local edge compute eliminates heavy cloud server emissions and optimizes ambulance routes.

**[4:25–4:50] SLIDE 10: ROADMAP & SCALABILITY**  
• **2026:** School zones and public square pilot.  
• **2027:** Citywide expansion with direct API dispatch to Protection Civile.  
• **2028–2030:** Expansion across Morocco and export to budget-constrained African smart cities.

**[4:50–5:00] SLIDE 11: CONCLUSION**  
Smart cities are not defined by how many expensive cameras they buy, but by how intelligently they use them to save lives. Smart Sahara proves Laâyoune can lead Africa in sovereign, frugal innovation. Thank you!
