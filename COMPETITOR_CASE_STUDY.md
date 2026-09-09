# Competitor case study: Staqu, Agrex AI, Wobot

Prepared 2026-09-08 for SentinelAI commercialisation. Figures are from public sources
(company sites, press interviews, Tracxn, CB Insights, PitchBook). Self-reported claims
are marked as such. No vendor publishes GPU sizing or independent accuracy audits.

## 1. Side by side

| | Staqu (JARVIS) | Agrex AI (AIVIS) | Wobot.ai |
|---|---|---|---|
| Founded / base | 2015, Gurugram | 2016, Gurugram | 2017, Delhi + Wilmington US |
| Funding | ~$2.1M total; INR 11 Cr pre-Series A (2022) led by Mount Judi, SIS | ~$407K seed | $5.4M to 5.8M; Peak XV Surge (Oct 2024), Titan Capital |
| Revenue | ARR INR 2.5 Cr (2022) to INR 40 Cr (2025), self-reported | not disclosed | third-party estimates $11M to 15M, unreliable |
| Team | not disclosed | small | ~110 (mid 2026) |
| Pricing | INR 2,000 to 8,000 per camera per month (press interview, Jan 2026); quote only on site | INR 800 to 1,200 per camera per month; 30-day free pilot | per location, from $40/location/month (2022 help doc), free tier 10 cameras; now custom quotes |
| Customers | 200+ clients claimed; Raymond (1,400 stores), JK Cement, Tata Projects, WeWork, GMR; 11 state police, 70 UP prisons (700 cameras, 2019), Gurugram Police ANPR | Bata, Pantaloons, M&S India; claims 100+ enterprises / 150k cameras (another page says 60+) | IRCTC, Rebel Foods, CureFit, Kitopi, Smartworks; US QSR chains via partners |
| Vertical | broad: security, retail, manufacturing, smart city, policing | retail only: footfall, queues, conversion, shrinkage | food service and QSR compliance: gloves, hairnets, temperature, checklists |
| Deployment | cloud (AWS, GCP), on-prem, edge; "data stays on customer infra"; 1 Mbps per camera; streaming agent removes need for static IP | cloud SaaS or edge box per site, no raw video egress | cloud SaaS on existing cameras; Microsoft Marketplace; hardware partners (Supermicro) |
| Face recognition | yes, 900k criminal records database for police (TRINETRA, Crime GPT) | no, explicitly anonymised | no |
| Certifications | ISO 27001, ISO 9001, SOC 2 Type II, GDPR | not published | G2 badges, no ISO published |
| Differentiator | government relationships, ~1M violence videos from UP Police as training data, audio analytics, LLM query layer (Jarvis GPT, WhatsApp) | retail ROI framing (conversion uplift %), 200+ retail models, agentic alerts | workflow product (checklists, audits) with AI as input; channel partners Interface Systems, Jolt, PAR |
| Go to market | direct enterprise and government; 2025 INTIN partnership for SMB | direct to retail chains, pilot first | channel partners and marketplaces, US focus |
| Public weak spots | no independent accuracy data; opaque pricing; policing FRT draws civil-liberty scrutiny (IFF Panoptic) | tiny funding; inconsistent scale claims; no identity features by design | not a security product; India is secondary market; pricing moved to opaque |

Market anchors: cloud CCTV analytics in India is sold at INR 500 to 2,000 per camera per
month. Human guard monitoring costs INR 2,000 to 5,000 per camera per month in allocated
guard time. Global benchmark $3 to 15 per camera per month.

## 2. What each one proves

Staqu proves the per-camera subscription works in India at INR 2,000 to 8,000 for
enterprise and government, that hybrid cloud/edge is expected, and that 1 Mbps per camera is
the accepted bandwidth budget. Its revenue growth came after it stopped selling "AI" and
started selling named outcomes per vertical. Its moat is data and relationships, not models.

Agrex proves a small team with under $0.5M can hold national retail accounts by owning one
vertical, pricing low, and framing the pitch as revenue uplift rather than security. It also
proves that skipping face recognition is a valid privacy positioning.

Wobot proves that the product customers pay for is often the workflow around the AI
(checklists, audits, alerts routed to the right person), and that channel partners who
already sell to the customer can carry a video product. It also shows the cost of a US-first
strategy: 110 people and $5M raised.

## 3. Where SentinelAI fits

None of the three owns institutions: schools and colleges, hospitals, corporate campuses,
gated residential communities, construction sites. These sites share one shape: a gate
(ANPR plus owner face match), corridors (loitering, crowd, fights), and a known population
(teacher, student, resident, worker attendance by face). That is exactly the feature set
already built. Position as campus and site safety, not generic video analytics.

Pricing lands between Agrex and Staqu: INR 1,000 Basic, 2,000 Plus, 4,000 Pro per camera per
month, sold against guard cost. Homes are not served by any of the three and should wait
until the optimised pipeline makes INR 375 per camera profitable.

Table stakes copied from all three: works on existing cameras, onboarding in under an hour,
a streaming agent so the client needs no static IP (the gateway box), edge option for data-
sensitive clients, 30-day event retention, mobile alerts.

The open lane: nobody publishes accuracy or capacity numbers. Publishing measured FAR/FRR
per camera class and cameras per GPU, with a written DPDP consent process, is a credible
differentiator for institutional buyers who must answer to boards and parents.

Risks learned from Staqu: face recognition sold to police invites scrutiny. Keep face
features to private premises with written consent and signage.

## Sources

- Staqu pricing and bandwidth: Electronics For You interview, Jan 2026
- Staqu clients, revenue, deployment: staqu.com (self-reported); Inc42, Deccan Chronicle (UP prisons 2019); Biometric Update (Crime GPT); Tracxn, CB Insights (funding)
- Agrex: agrexai.com (self-reported); Tracxn (funding)
- Wobot: help.wobot.ai (pricing 2022); Tracxn, PitchBook, CB Insights (funding); Interface Systems press release (2024); Latka, Growjo (revenue estimates)
- Market anchors: Wavestore 2026 ROI guide; Agrex India pricing guide
