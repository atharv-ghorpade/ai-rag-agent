"""
Script to populate data/parsed_cache/ with structured page-level text
for all newly provided PDFs and documents from the user prompt.
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data" / "parsed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_reflection_jan2026():
    pages = [
        {
            "page_num": 1,
            "text": (
                "DEPARTMENT OF INFORMATION TECHNOLOGY\n"
                "REFLECTION - NEWSLETTER\n"
                "JANUARY 2026 • ISSUE 02 • VOLUME XVI\n\n"
                "MESSAGE FROM THE HEAD OF DEPARTMENT\n"
                "By Prof. Dr. Vidya Chitre, Head of Department, Information Technology\n"
                "Welcome to the January 2026 edition (Issue 02, Volume XVI) of Reflection. Our department focuses on "
                "academic excellence, innovation, critical thinking, technical excellence, and preparing students to become future-ready professionals. "
                "We emphasize continuous adaptability, leadership, and ethical values in technology.\n\n"
                "VISION OF THE DEPARTMENT\n"
                "To be recognized as a centre of excellence in Information Technology where learners are nurtured "
                "to become competent professionals who benefit society.\n\n"
                "MISSION OF THE DEPARTMENT\n"
                "1. To evolve a curriculum emphasizing strong engineering fundamentals with flexibility to choose advanced courses "
                "and gain exposure to Information Technology tools and techniques.\n"
                "2. Encourage a teaching-learning process in which highly competent faculty share a symbiotic association with the institutes of repute.\n"
                "3. Facilitate creation and dissemination of knowledge through a digitally-enabled learning environment.\n"
                "4. Develop academic and infrastructural facilities with modern equipment and other learning resources and encourage reciprocal sharing with other institutes through networking.\n"
                "5. Establish a centre of excellence to enhance academia - industry partnerships and work on collaborative projects.\n\n"
                "PROGRAMME EDUCATIONAL OBJECTIVES (PEO)\n"
                "PEO 1: To provide students with strong foundational and technical knowledge in Information Technology.\n"
                "PEO 2: Graduates should analyze real-life problems and design technically sound, economically feasible and socially acceptable solutions, engaging in entrepreneurship.\n"
                "PEO 3: Graduates should exhibit professionalism, ethics, teamwork, social awareness, and lifelong learning to continuously improve professionally and technically.\n"
                "Primary objective of the department is to develop competent, ethical, and industry-ready professionals through quality education, research, and innovation."
            )
        },
        {
            "page_num": 2,
            "text": (
                "FACULTY RESEARCH & ACHIEVEMENTS\n\n"
                "The Department of Information Technology proudly highlights faculty research milestones:\n"
                "1. Prof. Rasika Ransing successfully completed her PhD in Natural Language Processing (NLP).\n"
                "2. Prof. Neha Kudu successfully completed her PhD in Artificial Intelligence and Machine Learning (AI & ML).\n\n"
                "These research accomplishments demonstrate academic excellence, research capability, and high expertise in emerging technologies "
                "including Natural Language Processing, Artificial Intelligence, and Machine Learning.\n"
                "Note on departmental research scope: The department emphasizes NLP and AI/ML; there are currently no faculty research publications "
                "or programs in Blockchain research, cybersecurity research, or cloud computing certifications in this issue."
            )
        }
    ]
    cache_path = CACHE_DIR / "Reflection_Jan2026.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")


def build_institute_level_admissions():
    pages = [
        {
            "page_num": 1,
            "text": (
                "VIDYALANKAR INSTITUTE OF TECHNOLOGY (VIT), WADALA (EAST), MUMBAI - 400037\n"
                "INSTITUTE LEVEL & AGAINST CAP (ACAP) ADMISSIONS POLICY & GUIDELINES\n\n"
                "1. Admission Categories:\n"
                "Institute Level admissions are divided into two primary categories:\n"
                "a) Against CAP (ACAP): Seats remaining vacant after the Centralized Admission Process (CAP Round IV) conducted by the State CET Cell.\n"
                "b) Institute Level (IL): Up to a maximum quota of 20% of the sanctioned intake of the program.\n\n"
                "2. Eligibility Criteria:\n"
                "Candidates who have registered online, completed document verification, and received confirmation from a registered Scrutiny Center (SC) are eligible for Institute Level and ACAP admissions.\n\n"
                "3. Accepted Entrance Examinations for MMS Admissions:\n"
                "Eligible entrance exams include MAH-MBA/MMS-CET, CAT, CMAT, XAT, ATMA, MAT, and GMAT. Submission of valid scorecard is mandatory."
            )
        },
        {
            "page_num": 2,
            "text": (
                "INSTITUTE LEVEL ADMISSION FEES AND PAYMENT STRUCTURE\n\n"
                "For Master of Management Studies (MMS) Institute Level and ACAP Admissions:\n"
                "- Tuition Fee: ₹1,73,479\n"
                "- Development Fee: ₹26,021\n"
                "- Total Fee Payable: ₹1,99,500\n\n"
                "Fee Equality Rule: Both ACAP and Institute Level admissions have the exact same prescribed fee of ₹1,99,500. ACAP students do not pay less.\n"
                "Payment Terms:\n"
                "- Cheque payments are NOT accepted.\n"
                "- Accepted Payment Methods: Demand Draft (DD), Pay Order, Credit Card, Debit Card, or Bank Transfer (NEFT/RTGS).\n"
                "- Demand Draft Favored Name: Demand Draft should be made in favor of 'Vidyalankar Institute of Technology'.\n"
                "- Designate Bank: Axis Bank. Candidates are advised to complete beneficiary registration at least one day before reporting.\n"
                "- Mandatory Fee Condition: Full fee payment is mandatory for admission confirmation. Failure to pay the full fees results in automatic cancellation of the allotted seat."
            )
        },
        {
            "page_num": 3,
            "text": (
                "ADMISSION MERIT LIST, COUNSELLING & GRIEVANCE PROCEDURE\n\n"
                "1. Application Process: The first step is submitting the online Institute Level application form on the institute portal.\n"
                "2. Merit List Preparation: Merging applications received through State CET Cell and direct institute applications.\n"
                "3. Publication: Merit list and vacancy details are displayed on the institute website (www.vit.edu.in) and displayed on the institute notice board.\n"
                "4. Allotment: Admissions are allotted through in-person counselling in the strict order of inter-se merit.\n"
                "5. Late Reporting: Candidates reporting late may only be offered seats based on residual availability.\n"
                "6. Grievance Redressal: Candidates can raise objections to the provisional merit list during the specified grievance period by submitting emails to the designated admission email. The final merit list is published after resolving valid grievances.\n"
                "7. Mandatory Documents: CET verification acknowledgement, allotment letter, entrance exam scorecard (MAH-CET/CAT/CMAT), SSC and HSC marksheets, degree mark sheets, passing certificates, transfer certificate, nationality certificate. Missing documents may lead to cancellation."
            )
        }
    ]
    cache_path = CACHE_DIR / "Institute_Level_Admissions_Policy.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")


def build_bytes_jan2026():
    pages = [
        {
            "page_num": 1,
            "text": (
                "BYTES - DEPARTMENT OF COMPUTER ENGINEERING\n"
                "JANUARY 2026 • ISSUE 02 • VOLUME 17\n"
                "Vidyalankar Institute of Technology (An Autonomous Institute Affiliated to University of Mumbai)\n"
                "Head of Department Message: Dr. Ravindra Sangle, Head of Department, Computer Engineering.\n"
                "This edition of BYTES reflects the dynamic progress of the Department of Computer Engineering through academic rigor, "
                "research excellence, and student development. Student accomplishments across CESA and CSI-VIT, technical events, podcasts, and alumni mentorship.\n"
                "Vision of the Department: To be recognized as a Centre of Excellence in the field of Computer Engineering where learners are nurtured in scholarly environment to evolve into competent Computer Engineering professionals to benefit society.\n"
                "Mission of the Department: 1. Evolve a curriculum emphasizing strong engineering fundamentals with flexibility for advanced courses. 2. Encourage faculty collaboration with reputed institutes. 3. Facilitate digital learning. 4. Modern equipment and networking. 5. Academia-industry partnerships.\n"
                "PEO: 1. Knowledge and employability. 2. Research and entrepreneurship. 3. Ethics, teamwork, and social awareness."
            )
        },
        {
            "page_num": 2,
            "text": (
                "STUDENT APPRECIATION & TiH GRANT\n"
                "Student Appreciation: Mr. Tejas Pradip Pawar from second year of Computer Engineering secured SECOND RANK at the 20th Aavishkar - Inter-Collegiate Research Convention (AY 2025-26) under mentorship of Dr. Amit K. Nerurkar.\n"
                "TiH Grant: The TIH Foundation at IIT Bombay (under NM-ICPS) CHANAKYA Fellowship Program awarded three students from TE CMPN A: Ms. Devanshi Mahajan, Ms. Sakshi Patil (guided by Dr. Amit K. Nerurkar), and Mr. Sheshank Singh (guided by Dr. Sheetal M.). "
                "Fellowship: ₹10,000 per month (₹1,00,000 over 10 months). Project: Image-based crop disease detection system leveraging AI/ML techniques and computer vision.\n"
                "Department Staff: Dr. Mandar Sohani (Professor, PhD Computer Engineering, 25 years experience, Embedded Systems, IoT, Networking). Dr. Umesh Kulkarni (Professor, PhD Computer Engineering, 28 years experience, AI, Data Science, Blockchain, Distributed Systems)."
            )
        },
        {
            "page_num": 3,
            "text": (
                "STUDENT'S SPEAK: DEVANSHI MAHAJAN (TE CMPN A)\n"
                "Devanshi Mahajan served as Class Representative and Documentation Head of CESA under Dr. Suja Jayachandran. Mentored by Dr. Amit Nerurkar for the TIH Fellowship. Gained exposure through CDAC credit transfer course in advanced technologies, Automation Expo, and CSI/CESA technical events."
            )
        },
        {
            "page_num": 4,
            "text": (
                "KNOW AN ALUMNUS: DEEP SHAHANE & CSI PODCAST\n"
                "Deep Shahane: Alumnus of VIT Computer Engineering, MS in Computer Science at University at Buffalo, Research Assistant at SAIR Lab. Mentored by Dr. Amit Nerurkar.\n"
                "CSI Event Podcast on 'Data Beyond Dashboard': Hosted by Ms. Gautami Kamble (CSI Publicity Head) featuring guest speaker Ms. Rujuta Lanke (Senior Data Analyst at American Express). Focused on data analytics and modern business strategies."
            )
        },
        {
            "page_num": 5,
            "text": (
                "CESA EVENT: COGNITION CLASH (14th October 2025)\n"
                "Organized by CESA. Evaluated AI/ML models in two phases: Pitch & Prototype Demonstration and Judicial Cross-Questioning on dataset size, overfitting, model selection, latency vs accuracy trade-offs, and scalability for 10,000 users."
            )
        },
        {
            "page_num": 6,
            "text": (
                "TECH COGNISPHERE 2025 & CSI-VIT EXPERT LECTURE\n"
                "Tech Cognisphere 2025 (4th October 2025): Organized by CSI VIT, TSEC CodeTantra, and CESA on UN 17 SDGs and sustainable technology. Hosted by Dr. Amit K. Nerurkar.\n"
                "CSI-VIT Expert Lecture (17th October 2025): 'Scalable Anomaly Detection Across Distributed Architectures' delivered by Ms. Ridhima Tambde (Data Scientist - Bot & Abuse) at VIT Auditorium."
            )
        },
        {
            "page_num": 7,
            "text": (
                "STUDENT ARTICLE BY SIDDHI GAWADE: MEMORY-FIRST COMPUTING\n"
                "Article: 'Memory-First Computing: Why Processing Is Moving Toward Data' by Siddhi Gawade (TE-A). Analyzes near-memory computing, in-memory computing, and memory-centric computing for AI training, edge devices, and cloud computing bottlenecks."
            )
        },
        {
            "page_num": 8,
            "text": (
                "UPCOMING EVENTS & EDITORIAL TEAM\n"
                "- Orientation Program for Even Sem 25-26 in January 2026\n"
                "- CSI-VIT Flagship Event ENTHUSIA in March 2026\n"
                "- CESA-VIT Flagship Event PLETHORA in February 2026\n"
                "- VERVE 26 in February 2026\n"
                "- TEDxVIT in March 2026\n"
                "Editorial Team: Dr. Amit K. Nerurkar (Chief Editor). Website: www.vit.edu.in"
            )
        }
    ]
    cache_path = CACHE_DIR / "BYTES_Jan2026_Vol17_Issue02.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")


def build_bytes_july2026():
    pages = [
        {
            "page_num": 1,
            "text": (
                "BYTES - DEPARTMENT OF COMPUTER ENGINEERING\n"
                "JULY 2026 • ISSUE 04 • VOLUME 17\n"
                "Vidyalankar Institute of Technology (An Autonomous Institute Affiliated to University of Mumbai)\n"
                "Head of Department Message: Dr. Ravindra Sangle, Head of Department, Computer Engineering.\n"
                "Highlights academic excellence, research contributions, and student activities embracing AI, Cybersecurity, and Cloud Computing.\n"
                "Vision & Mission: Centre of Excellence in Computer Engineering, curriculum innovation, research, digital learning, and industry partnerships."
            )
        },
        {
            "page_num": 2,
            "text": (
                "DEPARTMENTAL UPDATES & STAFF\n"
                "Faculty Achievements:\n"
                "- Dr. Suja Jayachandran: Expert session on 'Large Language Models in Clinical Applications' at ISTE FDP. Reviewer for Scientific Reports, Discover AI, Computers & Education: AI. Completed AI-Driven Software Development STTP (SVNIT) and NPTEL Soft Computing.\n"
                "- Dr. Suvarna Bhat: Session Chair at ICSSSD 2026; Reviewer for AI Review, SIVP, BMC Oral Health; Editorial Board Academic Trends & Innovative Research; Workshop at VIT Vellore.\n"
                "- Mr. Mahesh Ashok Khandke: Research paper on satellite image deblurring using Deep Learning and Walrus Optimization at ICICGR 2026. STTP on AI-driven software development.\n"
                "- Dr. Amit K. Nerurkar, Dr. Swapnil Sonawane, Dr. Sanjeev Dwivedi, Dr. Ravindra Sangle: 1 Month Internship in AI-Driven Software Development (Pantech & SVNIT Surat).\n"
                "- Dr. Swapnil Sonawane: Session Chair at ICICT 2026.\n"
                "- Dr. Amit K. Nerurkar: Elite certification (top 2% nationwide) in 12-week NPTEL Computer Networks and IP.\n"
                "- Best Paper Award: Dr. Ravindra Sangle, Dr. Amit K. Nerurkar, Dr. Swapnil Sonawane, Dr. Sanjeev Dwivedi presented 'A Review of Secure Health Tracking Using Smart Technology' winning Best Paper Presentation (Virtual) at 9th ICMDRSE-2026, Kuala Lumpur, Malaysia.\n"
                "Department Staff: Mrs. Sonali Surve (Executive, B.Com, 9 yrs exp, Advanced Excel, MS-Office, Google Workspace). Ms. Sayli Patil (Executive, BMS, 1 yr exp, Tally ERP 9, GST)."
            )
        },
        {
            "page_num": 3,
            "text": (
                "STUDENT'S SPEAK: MINAKSHI SAHOO (SE-B)\n"
                "Reflects on joining in September 2024. Mentored by Dr. Ravindra Sangle, Dr. Amit Nerurkar (CSI VIT), and Dr. Sachin Deshpande (TEDxVIT). Credit Transfer Courses in summer break after sem 4 provided industry skills."
            )
        },
        {
            "page_num": 4,
            "text": (
                "KNOW AN ALUMNUS: SRISHTI AGRAWAL (CMPN 2017-2021)\n"
                "PhD student in Human-Computer Interaction (HCI) at Purdue University, Doctoral Researcher at DVC Lab focusing on human-AI trust calibration. Published at CHI 2025 (Yokohama), NeurIPS 2024, ASSETS 2023, IDC 2023. Worked at Military Family Research Institute and NASA Mars mission. Mentored by Dr. Varsha Bhosle, Dr. Dilip Motwani, and Dr. Amit Nerurkar."
            )
        },
        {
            "page_num": 5,
            "text": (
                "GDG-VIT EVENTS (APRIL 2026)\n"
                "1. The Last Standing Ronin (April 1, 2026): Anime-themed coding contest featuring DSA and rapid problem-solving.\n"
                "2. The Age of AI Agents (April 2, 2026): Speaker session on Agentic AI, autonomous workflows, and industrial applications.\n"
                "3. The Way of Ghost: Bluff & Bid (April 3, 2026): Coding, logic, and strategic decision-making competition."
            )
        },
        {
            "page_num": 6,
            "text": (
                "GDG HACKATHON & CSI-VIT ENTHUSIA\n"
                "HackBuild 2026 12-Hour Hackathon (April 4, 2026): GDG On Campus VIT & GDG UMIT collaboration for functional prototyping.\n"
                "CSI-VIT ENTHUSIA Flagship (March-April 2026):\n"
                "- Design Paradox: UI/UX and product design challenge.\n"
                "- MindMare: Strategic thinking and high-pressure decision challenges.\n"
                "- Parallax: Hacker-escape puzzle room experience.\n"
                "- Grid Rush: Sudoku-based algorithmic coding efficiency competition."
            )
        },
        {
            "page_num": 7,
            "text": (
                "STUDENT ARTICLE: SOUMITRA RAJGURU\n"
                "Article: 'AI Is Not the Bubble, LLMs Might Be: Are We Mistaking LLMs for AI?' by Soumitra Rajguru (SE C). "
                "Explores distinction between broad AI (computer vision, robotics, reinforcement learning) and LLMs as a visible interface. Compares current LLM adoption to dot-com era."
            )
        },
        {
            "page_num": 8,
            "text": (
                "UPCOMING EVENTS & EDITORIAL TEAM\n"
                "- FDP on Cyber Security and Data Protection: Understanding DPDP Act in July 2026.\n"
                "- Orientation Programme for Odd Sem 2026-27 in July 2026.\n"
                "- CSI-VIT Emerging Trends Podcast in July 2026.\n"
                "- CSI-VIT CXO Meet with Cyberfrat in August 2026.\n"
                "Editorial Team: Dr. Amit K. Nerurkar (Chief Editor). Website: www.vit.edu.in"
            )
        }
    ]
    cache_path = CACHE_DIR / "BYTES_July2026_Vol17_Issue04.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")


def build_curriculum_caches():
    # Honours/Minor R-2022
    hm_pages = [
        {
            "page_num": 1,
            "text": (
                "VIDYALANKAR INSTITUTE OF TECHNOLOGY (AUTONOMOUS - AFFILIATED TO UNIVERSITY OF MUMBAI)\n"
                "HONOURS / MINOR DEGREE PROGRAMME FOR B.TECH IN COMPUTER ENGINEERING (R-2022 CURRICULUM)\n"
                "Preamble: Autonomous Choice-Based Credit and Grading System (CBCGS) as per AICTE model curriculum. "
                "Offers academic flexibility, industry-aligned tracks, and add-on Honours/Minor degree requiring additional 18 credits."
            )
        },
        {
            "page_num": 2,
            "text": (
                "GUIDELINES FOR AWARD OF HONOURS / MINOR DEGREE PROGRAMME\n"
                "- Honours/Minor requires 18 credits over and above minimum B.Tech degree credits.\n"
                "- Opting period: Learners opt during the break of Semester 5 and Semester 6.\n"
                "- Eligibility: Minimum CGPA of 6.75 at the end of 4th semester and earned 80 credits from Sem 1 to Sem 4 (41 credits for DSY students).\n"
                "- Credit transfer allowed if relevant courses completed previously."
            )
        },
        {
            "page_num": 3,
            "text": (
                "SYLLABUS SCHEME TEMPLATE FOR HONOURS/MINOR DEGREE (18 CREDITS TOTAL)\n"
                "- HM01: Industry Interaction (Theory, Break of Sem5 & Sem6, 1 credit, 25 marks)\n"
                "- Course 1: Theory (Sem 6, 2 credits, 75 marks) + Lab (Sem 6, 1 credit, 50 marks)\n"
                "- HM02: Survey Report/Paper (Break of Sem6 & Sem7, 2 credits, 50 marks)\n"
                "- Course 2: Theory (Sem 7, 2 credits, 75 marks) + Lab (Sem 7, 1 credit, 50 marks)\n"
                "- HM03: Seminar (Break of Sem7 & Sem8, 2 credits, 50 marks)\n"
                "- Course 3: Theory (Sem 8, 2 credits, 75 marks) + Lab (Sem 8, 1 credit, 50 marks)\n"
                "- HM04: Capstone Project (Sem 8, 4 credits, 125 marks)"
            )
        },
        {
            "page_num": 4,
            "text": (
                "HONOURS/MINOR PROGRAMMES OFFERED\n"
                "1. Next-Gen Artificial Intelligence & Machine Learning (Dept of Computer Engg): Ethics in AI (HMCE01T/P), Scalable ML and BDA (HMCE02T/P), Generative AI models (HMCE03T/P).\n"
                "2. Next-Gen Data Science (Dept of Computer Engg): Responsible AI and Data Ethics (HMCE04T/P), Cloud native & BDA (HMCE05T/P), Realtime Data Analytics (HMCE06T/P).\n"
                "3. Artificial Intelligence and Machine Learning: Soft Computing (HMCE07T/P), Natural Language Processing (HMCE08T/P), Deep Learning (HMCE09T/P).\n"
                "4. Data Science: Advanced Databases (HMCE10T/P), Big Data Analytics (HMCE11T/P), Text, Web & Social Media Analytics (HMCE12T/P).\n"
                "5. Cyber Security: Cryptography and Network Security (HMCE13T/P), Web Application Security (HMCE14T/P), Digital Forensics (HMCE15T/P)."
            )
        }
    ]
    cache_path = CACHE_DIR / "BTech_Computer_Engineering_Honours_Minor_R2022.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(hm_pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")

    # Final Year R-2023
    fy_pages = [
        {
            "page_num": 1,
            "text": (
                "VIDYALANKAR INSTITUTE OF TECHNOLOGY\n"
                "BACHELOR OF TECHNOLOGY IN COMPUTER ENGINEERING\n"
                "FINAL YEAR SCHEME & SYLLABUS (R-2023) - NEP ALIGNED\n"
                "Preamble: NEP framework enabling academic flexibility, experiential learning, and multidisciplinary education."
            )
        },
        {
            "page_num": 2,
            "text": (
                "SEMESTER VII SCHEME (TOTAL 21 CREDITS):\n"
                "- PCCE17: Programming with R (1 credit, Practical)\n"
                "- PCCE19: DevOps (2 credits, Theory + Practical)\n"
                "- Programme Elective 4 (3 credits): NLP (PECE05T/P), Text Web & Social Media Analytics (PECE07T/P), IoT & Edge Computing (PECE17T/P), Web Application Security (PECE12T/P)\n"
                "- Programme Elective 5 (3 credits): Advance ML (PECE09T/P), Big Data Analytics (PECE14T/P), IoT Security & Trust (PECE19T/P), Malware Analysis (PECE15T/P)\n"
                "- Programme Elective 6 (3 credits): Deep Learning (PECE13T/P), Recommendation System (PECE18T/P), Industrial IoT (PECE20T/P), Mobile and Wireless Security (PECE16T/P)\n"
                "- PRJCE03: Project-2 Final (4 credits)\n"
                "- MDM02: Skill & Competency Development 2 (1 credit)\n"
                "- MDMXX: Multidisciplinary Minor Course 3 (4 credits): Bioinformatics (MDMBI03), Scaling Startups (MDMIE03), Strategic Marketing (MDMBD03), Intelligent Mobile Robotics (MDMRB03)\n"
                "- OJT01: Industry Internship 1 (5 credits, 150+ hours completed in break)"
            )
        },
        {
            "page_num": 3,
            "text": (
                "SEMESTER VIII SCHEME (TOTAL 12 CREDITS):\n"
                "- OJT02: Industry Internship 2 (7 credits, 200 marks)\n"
                "- RM01: Research Methodology (3 credits, 100 marks)\n"
                "- Open Elective 2 (2 credits): Cyber Law (OEC02), Project Management (OEC03), Product Lifecycle Management (OEC04), Sustainability Management (OEC05), Renewable Energy Management (OEC06), Digital Public Infrastructure (OEC16)"
            )
        }
    ]
    cache_path = CACHE_DIR / "BTech_Computer_Engineering_Final_Year_Syllabus_R2023.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(fy_pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")

    # Third Year R-2023
    ty_pages = [
        {
            "page_num": 1,
            "text": (
                "VIDYALANKAR INSTITUTE OF TECHNOLOGY\n"
                "BACHELOR OF TECHNOLOGY IN COMPUTER ENGINEERING WITH MULTIDISCIPLINARY MINOR\n"
                "THIRD YEAR SCHEME & SYLLABUS (R-2023) - NEP ALIGNED\n"
                "Preamble: Autonomous curriculum with PCC, PEC, OEC, MDM, ELC, and Value Education Courses (VEC)."
            )
        },
        {
            "page_num": 2,
            "text": (
                "SEMESTER V SCHEME (TOTAL 20 CREDITS):\n"
                "- PCCE10T/P: Artificial Intelligence (Theory + Lab, 3 credits)\n"
                "- PCCE12T/P: Software Engineering & Web Design Lab (3 credits)\n"
                "- PCCE14: Cloud Computing Lab (1 credit)\n"
                "- PCCE15: System Programming and Compiler Design (3 credits)\n"
                "- PCCE16T/P: Distributed Systems (Theory + Lab, 3 credits)\n"
                "- Programme Elective 1 (3 credits): Data Warehousing and Data Mining (PECE02T/P), Modern Sensors for IoT (PECE03T/P), Computer and Network Security (PECE04T/P)\n"
                "- MDM Course 1 (4 credits): Bioinformatics (MDMBI01), Foundations of Innovation (MDMIE01), Business Development (MDMBD01), Fundamentals of Robotics (MDMRB01)"
            )
        },
        {
            "page_num": 3,
            "text": (
                "SEMESTER VI SCHEME (TOTAL 20 CREDITS):\n"
                "- PCCE13T/P: Machine Learning (Theory + Lab, 3 credits)\n"
                "- PCCE18: Machine Vision using Python (1 credit)\n"
                "- Value Education Course (VEC, 2 credits): E-waste (VEC02), Universal Human Values (VEC03), Sustainable Environment (VEC04), Four Pillars of Democratic Nation (VEC05)\n"
                "- Programme Elective 2 (3 credits): Soft Computing (PECE01T/P), Advanced Databases (PECE06T/P), Embedded Systems and Tiny OS (PECE11T/P), System Security and Ethical Hacking (PECE08T/P)\n"
                "- Programme Elective 3 (3 credits): Probabilistic and Graphical Models (PECE10), Principles of IoT (PECE22T/P), Digital Forensics (PECE21T/P)\n"
                "- PRJCE01: Specialization-Based Project (2 credits)\n"
                "- PRJCE02: Project-1 Synopsis (2 credits)\n"
                "- MDM Course 2 (4 credits): Algorithms in Bioinformatics (MDMBI02), Startup Planning (MDMIE02), Financial Basics (MDMBD02), Machine Vision & Robotic Perception (MDMRB02)"
            )
        }
    ]
    cache_path = CACHE_DIR / "BTech_Computer_Engineering_Third_Year_Syllabus_R2023.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ty_pages, f, ensure_ascii=False, indent=2)
    print(f"Created: {cache_path}")


if __name__ == "__main__":
    print("Building parsed caches for new documents...")
    build_reflection_jan2026()
    build_institute_level_admissions()
    build_bytes_jan2026()
    build_bytes_july2026()
    build_curriculum_caches()
    print("All document caches successfully populated!")
