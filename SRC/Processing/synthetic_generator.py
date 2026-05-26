"""
Sinh dữ liệu CV tổng hợp cho các class thiếu dữ liệu.
Mỗi mẫu được tạo từ template + keyword pool ngẫu nhiên → văn bản tự nhiên.
Output: Data/Raw/synthetic_resumes.csv  (cùng cấu trúc Resume.csv)
"""

import random
import os
import pandas as pd

random.seed(42)

# ── Keyword pools theo từng ngành ──────────────────────────────────────────────
POOLS = {

    "AGRICULTURE": {
        "skills": [
            "crop management", "soil science", "irrigation systems", "agronomy",
            "horticulture", "pest control", "fertilizer application", "farm management",
            "precision agriculture", "livestock management", "greenhouse management",
            "seed selection", "harvest planning", "organic farming", "aquaculture",
            "plant pathology", "food safety", "supply chain", "gis mapping",
            "water resource management", "sustainable agriculture", "animal husbandry",
        ],
        "tools": [
            "GPS tractor guidance", "drone scouting", "soil testing kits",
            "irrigation controllers", "farm management software", "GIS", "NDVI sensors",
        ],
        "certs": [
            "BSc Agriculture", "MSc Agronomy", "Diploma in Horticulture",
            "BSc Food Technology", "Certified Crop Adviser (CCA)",
        ],
        "roles": [
            "farm manager", "agricultural officer", "crop scientist",
            "agronomist", "horticulturist", "field supervisor",
            "agriculture extension officer", "soil technician",
        ],
        "objectives": [
            "seeking a challenging role in agricultural management to apply knowledge of crop science",
            "aspiring to contribute to sustainable farming practices and food security",
            "looking for a position in agronomy to improve crop yield and farm profitability",
            "dedicated agriculture professional aiming to optimize farm operations",
        ],
    },

    "DIGITAL-MEDIA": {
        "skills": [
            "SEO", "SEM", "Google Analytics", "Facebook Ads", "Instagram marketing",
            "content creation", "email marketing", "social media management",
            "video production", "YouTube marketing", "TikTok strategy",
            "influencer marketing", "PPC campaigns", "brand strategy",
            "copywriting", "A/B testing", "conversion rate optimization",
            "affiliate marketing", "podcast production", "web analytics",
        ],
        "tools": [
            "Google Ads", "Meta Business Suite", "Hootsuite", "Canva",
            "Adobe Premiere", "MailChimp", "HubSpot", "SEMrush", "Ahrefs",
            "Buffer", "Sprout Social",
        ],
        "certs": [
            "Google Digital Marketing Certification", "HubSpot Content Marketing",
            "BSc Mass Communication", "BA Journalism", "MBA Marketing",
            "Facebook Blueprint Certification",
        ],
        "roles": [
            "digital marketing executive", "social media manager", "SEO specialist",
            "content strategist", "paid media analyst", "brand manager",
            "email marketing coordinator", "digital content producer",
        ],
        "objectives": [
            "creative digital marketing professional seeking to drive brand growth through data-driven campaigns",
            "passionate about leveraging social media and content to build engaged online communities",
            "results-oriented digital marketer with expertise in SEO and paid advertising",
            "looking to grow brand presence through innovative digital strategies",
        ],
    },

    "APPAREL": {
        "skills": [
            "fashion merchandising", "garment construction", "textile knowledge",
            "fashion buying", "retail planning", "visual merchandising",
            "trend forecasting", "pattern making", "product development",
            "supply chain management", "inventory management", "fashion design",
            "brand development", "fabric sourcing", "quality control",
            "e-commerce fashion", "style consulting", "collection planning",
        ],
        "tools": [
            "Adobe Illustrator", "CLO 3D", "Photoshop", "ERP systems",
            "retail management software", "CAD for fashion",
        ],
        "certs": [
            "BSc Fashion Design", "BA Textile Design", "Diploma in Apparel Management",
            "MBA Fashion Management", "Certificate in Fashion Merchandising",
        ],
        "roles": [
            "fashion designer", "apparel merchandiser", "garment technician",
            "fashion buyer", "retail fashion manager", "production coordinator",
            "textile quality inspector", "style consultant",
        ],
        "objectives": [
            "creative fashion professional with strong understanding of garment construction and market trends",
            "seeking a role in apparel merchandising to apply expertise in trend forecasting and buying",
            "passionate about sustainable fashion and ethical garment production",
            "dedicated apparel professional aiming to blend creativity with business acumen",
        ],
    },

    "ARTS": {
        "skills": [
            "fine arts", "oil painting", "watercolor", "sculpture",
            "art direction", "photography", "illustration", "printmaking",
            "ceramics", "digital art", "art curation", "portfolio management",
            "art education", "creative writing", "mixed media", "installation art",
            "art history", "exhibition planning", "cultural program management",
        ],
        "tools": [
            "Adobe Photoshop", "Procreate", "Lightroom", "InDesign",
            "traditional media", "darkroom photography", "ceramics wheel",
        ],
        "certs": [
            "BFA Fine Arts", "MFA Visual Arts", "BA Art History",
            "Diploma in Graphic Arts", "Certificate in Photography",
        ],
        "roles": [
            "visual artist", "art director", "illustrator", "photographer",
            "art teacher", "gallery curator", "creative director",
            "art educator", "multimedia artist",
        ],
        "objectives": [
            "passionate visual artist seeking to express creativity through diverse artistic mediums",
            "experienced artist and educator looking to inspire creativity in institutional settings",
            "creative professional with expertise in fine arts and digital illustration",
            "art curator dedicated to promoting emerging artists and contemporary art exhibitions",
        ],
    },

    "BPO": {
        "skills": [
            "customer service", "call center operations", "voice process",
            "non-voice process", "data entry", "back office support",
            "CRM management", "complaint resolution", "quality assurance",
            "team leadership", "SLA management", "KPI monitoring",
            "client relationship", "process improvement", "chat support",
            "email support", "ticketing systems", "workforce management",
        ],
        "tools": [
            "Salesforce", "Zendesk", "Freshdesk", "Avaya", "Genesys",
            "Microsoft Office", "JIRA", "ServiceNow",
        ],
        "certs": [
            "BA Business Administration", "BSc Commerce",
            "Diploma in Customer Service", "COPC Certification",
            "Six Sigma Green Belt",
        ],
        "roles": [
            "customer service representative", "call center agent", "BPO executive",
            "team leader", "quality analyst", "process associate",
            "back office executive", "customer support specialist",
        ],
        "objectives": [
            "experienced BPO professional with strong customer service and communication skills",
            "seeking a leadership role in business process outsourcing to drive team performance",
            "dedicated call center professional focused on improving customer satisfaction metrics",
            "result-driven BPO executive with expertise in managing high-volume customer interactions",
        ],
    },

    "DESIGNER": {
        "skills": [
            "UI design", "UX research", "wireframing", "prototyping",
            "graphic design", "branding", "typography", "color theory",
            "motion graphics", "web design", "print design", "logo design",
            "user testing", "design thinking", "interaction design",
            "visual design", "mobile app design", "design systems",
        ],
        "tools": [
            "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator",
            "InDesign", "After Effects", "Zeplin", "InVision",
        ],
        "certs": [
            "BDes Visual Communication", "BA Graphic Design",
            "Google UX Design Certificate", "Interaction Design Foundation",
            "Adobe Certified Professional",
        ],
        "roles": [
            "UI/UX designer", "graphic designer", "visual designer",
            "product designer", "brand designer", "web designer",
            "motion designer", "interaction designer",
        ],
        "objectives": [
            "creative UI/UX designer passionate about crafting intuitive and visually compelling user experiences",
            "graphic designer with strong branding and visual communication skills seeking creative roles",
            "user-centered product designer looking to bridge business goals with elegant design solutions",
            "skilled visual designer with expertise in digital and print media",
        ],
    },

    "HR": {
        "skills": [
            "talent acquisition", "recruitment", "onboarding", "payroll processing",
            "employee relations", "performance management", "HR policy",
            "training and development", "HRIS", "compensation and benefits",
            "organizational development", "labor law compliance",
            "workforce planning", "exit interviews", "HR analytics",
            "conflict resolution", "diversity and inclusion",
        ],
        "tools": [
            "SAP HR", "Workday", "BambooHR", "ATS systems", "LinkedIn Recruiter",
            "Zoho People", "PeopleSoft",
        ],
        "certs": [
            "MBA Human Resources", "BSc Psychology", "SHRM-CP",
            "HRCI Certification", "Diploma in HR Management",
        ],
        "roles": [
            "HR manager", "talent acquisition specialist", "HR business partner",
            "recruitment officer", "HR executive", "payroll specialist",
            "training coordinator", "people operations manager",
        ],
        "objectives": [
            "experienced HR professional dedicated to attracting top talent and fostering a positive work culture",
            "seeking an HR business partner role to align people strategy with organizational goals",
            "passionate about employee development and creating inclusive workplace environments",
            "HR specialist with expertise in end-to-end recruitment and talent management",
        ],
    },

    "CONSTRUCTION": {
        "skills": [
            "site management", "civil engineering", "structural design",
            "project planning", "quantity surveying", "AutoCAD",
            "blueprint reading", "budget estimation", "contract management",
            "safety compliance", "RCC design", "building codes",
            "subcontractor management", "progress reporting",
            "concrete technology", "surveying", "MEP coordination",
        ],
        "tools": [
            "AutoCAD", "Revit", "MS Project", "Primavera P6",
            "ETABS", "STAAD Pro", "SAP2000",
        ],
        "certs": [
            "BE Civil Engineering", "BSc Construction Management",
            "PMP Certification", "OSHA Safety Certification",
            "Diploma in Civil Engineering",
        ],
        "roles": [
            "site engineer", "project manager", "quantity surveyor",
            "structural engineer", "construction supervisor",
            "civil engineer", "planning engineer", "contracts manager",
        ],
        "objectives": [
            "civil engineer with strong site management and structural design expertise",
            "seeking a construction project management role to deliver high-quality infrastructure on time",
            "dedicated site engineer with expertise in RCC structures and safety compliance",
            "construction professional aiming to lead large-scale building projects efficiently",
        ],
    },

    "HEALTHCARE": {
        "skills": [
            "patient care", "clinical assessment", "medical records",
            "nursing", "vital signs monitoring", "medication administration",
            "emergency care", "health education", "wound management",
            "infection control", "medical terminology", "EHR systems",
            "diagnostic procedures", "patient advocacy", "rehabilitation",
            "health promotion", "triage", "phlebotomy",
        ],
        "tools": [
            "Epic EHR", "Meditech", "Cerner", "point-of-care testing",
            "medical imaging", "ECG machines",
        ],
        "certs": [
            "MBBS", "BSc Nursing", "MD Internal Medicine",
            "ACLS Certification", "BLS Certification",
            "Diploma in Medical Lab Technology",
        ],
        "roles": [
            "registered nurse", "physician", "clinical officer",
            "medical laboratory technician", "physiotherapist",
            "healthcare assistant", "ward nurse", "emergency nurse",
        ],
        "objectives": [
            "compassionate healthcare professional committed to delivering high-quality patient-centered care",
            "registered nurse seeking a challenging clinical role in a dynamic hospital environment",
            "dedicated physician with expertise in diagnosis and treatment of acute and chronic conditions",
            "healthcare worker passionate about improving patient outcomes and community health",
        ],
    },

    "FITNESS": {
        "skills": [
            "personal training", "fitness assessment", "exercise programming",
            "nutrition coaching", "group fitness instruction",
            "strength and conditioning", "yoga instruction", "sports coaching",
            "injury prevention", "body composition analysis", "cardio training",
            "weight management", "wellness coaching", "HIIT",
            "functional training", "client motivation", "sports nutrition",
        ],
        "tools": [
            "gym equipment operation", "fitness tracking apps",
            "body composition analyzers", "heart rate monitors",
            "MyFitnessPal", "Trainerize",
        ],
        "certs": [
            "BSc Sports Science", "NASM Certified Personal Trainer",
            "ACE Fitness Instructor", "Yoga Alliance RYT 200",
            "Diploma in Physical Education", "ACSM Certification",
        ],
        "roles": [
            "personal trainer", "fitness instructor", "gym manager",
            "sports coach", "yoga instructor", "wellness consultant",
            "strength and conditioning coach", "aerobics instructor",
        ],
        "objectives": [
            "certified personal trainer passionate about helping clients achieve their health and fitness goals",
            "fitness professional with expertise in strength training and nutrition guidance",
            "dedicated yoga instructor seeking to promote holistic health and mindfulness",
            "experienced sports coach committed to enhancing athletic performance and physical well-being",
        ],
    },

    "AVIATION": {
        "skills": [
            "flight operations", "aircraft navigation", "air traffic communication",
            "aviation safety", "flight planning", "instrument rating",
            "airline operations", "cabin crew management",
            "emergency procedures", "ICAO regulations", "DGCA compliance",
            "ground handling", "airport operations", "weather analysis",
            "crew resource management", "dispatch operations",
        ],
        "tools": [
            "flight simulators", "ACARS", "ATC communication systems",
            "Boeing 737 / Airbus A320", "flight management systems (FMS)",
            "NOTAM systems",
        ],
        "certs": [
            "Commercial Pilot License (CPL)", "ATPL",
            "BSc Aviation Management", "Cabin Crew Certification",
            "Airport Operations Certificate", "Dispatcher License",
        ],
        "roles": [
            "commercial pilot", "first officer", "cabin crew",
            "airport operations officer", "flight dispatcher",
            "ground handling supervisor", "aviation safety officer",
        ],
        "objectives": [
            "experienced commercial pilot with strong record in safe and on-time flight operations",
            "aviation professional dedicated to maintaining the highest standards of flight safety",
            "cabin crew member with excellent passenger service and emergency response skills",
            "seeking a challenging role in airline operations to leverage aviation management expertise",
        ],
    },

    "CHEF": {
        "skills": [
            "culinary arts", "menu planning", "food preparation",
            "kitchen management", "pastry and baking", "food safety",
            "HACCP", "inventory management", "sous vide",
            "catering management", "restaurant operations",
            "continental cuisine", "Indian cuisine", "Italian cooking",
            "staff supervision", "cost control", "food presentation",
        ],
        "tools": [
            "commercial kitchen equipment", "POS systems",
            "food thermometers", "specialty baking equipment",
        ],
        "certs": [
            "Diploma in Culinary Arts", "BSc Hotel Management",
            "Culinary Institute of America (CIA) Certificate",
            "ServSafe Food Handler", "WSET Wine Certificate",
        ],
        "roles": [
            "head chef", "sous chef", "pastry chef", "line cook",
            "executive chef", "catering manager", "kitchen supervisor",
            "restaurant manager",
        ],
        "objectives": [
            "passionate culinary professional with expertise in diverse cuisines and kitchen management",
            "creative chef seeking to design innovative menus that delight guests and drive restaurant success",
            "experienced sous chef ready to take on head chef responsibilities in a fine dining environment",
            "dedicated food service professional committed to high standards of culinary excellence",
        ],
    },

    "TEACHER": {
        "skills": [
            "lesson planning", "curriculum development", "classroom management",
            "student assessment", "differentiated instruction",
            "e-learning platforms", "parent communication",
            "special education", "STEM teaching", "language instruction",
            "educational psychology", "test preparation",
            "project-based learning", "blended learning", "mentoring",
        ],
        "tools": [
            "Google Classroom", "Zoom", "Moodle", "Khan Academy",
            "smart boards", "MS Teams for Education",
        ],
        "certs": [
            "BEd Education", "MA English", "BSc Mathematics",
            "CTET Certification", "MEd Curriculum and Instruction",
            "TEFL / TESOL Certificate",
        ],
        "roles": [
            "high school teacher", "primary school teacher",
            "university lecturer", "special education teacher",
            "English language instructor", "mathematics teacher",
            "science teacher", "academic coordinator",
        ],
        "objectives": [
            "passionate educator dedicated to fostering a love of learning in students of all abilities",
            "experienced teacher seeking to create inclusive and engaging classroom environments",
            "academic professional with expertise in curriculum design and student-centered pedagogy",
            "committed to nurturing students' academic and personal development",
        ],
    },

    "CONSULTANT": {
        "skills": [
            "management consulting", "business strategy", "process improvement",
            "stakeholder management", "project management", "change management",
            "financial analysis", "market research", "ERP implementation",
            "organizational restructuring", "risk assessment",
            "data analytics", "client relationship management",
            "SAP consulting", "supply chain optimization",
        ],
        "tools": [
            "SAP", "Oracle ERP", "MS Project", "Power BI",
            "Tableau", "Salesforce", "Excel advanced analytics",
        ],
        "certs": [
            "MBA Strategy", "PMP Certification", "CFA",
            "SAP Certified Consultant", "PRINCE2",
            "Six Sigma Black Belt",
        ],
        "roles": [
            "management consultant", "strategy consultant",
            "business analyst", "ERP consultant", "IT consultant",
            "change management advisor", "operations consultant",
        ],
        "objectives": [
            "strategic management consultant with a track record of driving business transformation",
            "results-driven consultant specializing in operational efficiency and organizational change",
            "seeking a senior consulting role to help organizations achieve sustainable competitive advantage",
            "analytical consultant with expertise in ERP implementation and process optimization",
        ],
    },

    "ADVOCATE": {
        "skills": [
            "legal research", "litigation", "contract drafting",
            "client counseling", "court representation",
            "criminal law", "civil litigation", "corporate law",
            "intellectual property", "legal documentation",
            "negotiation", "arbitration", "compliance",
            "due diligence", "legal advisory",
        ],
        "tools": [
            "LexisNexis", "Westlaw", "legal drafting software",
            "case management systems", "e-discovery tools",
        ],
        "certs": [
            "LLB Law", "LLM Corporate Law", "Bar Council enrollment",
            "Diploma in Cyber Law", "Certified Compliance Professional",
        ],
        "roles": [
            "advocate", "corporate lawyer", "legal consultant",
            "litigation attorney", "compliance officer",
            "in-house counsel", "paralegal",
        ],
        "objectives": [
            "experienced advocate with strong litigation and contract law expertise",
            "corporate lawyer seeking to provide strategic legal counsel to businesses",
            "dedicated legal professional committed to upholding justice and client rights",
            "compliance-focused attorney with expertise in regulatory frameworks",
        ],
    },

    "BUSINESS-DEVELOPMENT": {
        "skills": [
            "business development", "lead generation", "market expansion",
            "partnership building", "B2B sales", "proposal writing",
            "strategic planning", "CRM management", "revenue growth",
            "client acquisition", "competitive analysis",
            "pricing strategy", "contract negotiation", "KPI tracking",
        ],
        "tools": [
            "Salesforce", "HubSpot", "LinkedIn Sales Navigator",
            "Zoho CRM", "Tableau", "MS Excel",
        ],
        "certs": [
            "MBA Business Development", "BSc Business Administration",
            "Certified Business Development Professional",
            "PMP", "Dale Carnegie Sales Training",
        ],
        "roles": [
            "business development manager", "BD executive",
            "strategic partnerships manager", "growth manager",
            "market development representative", "account executive",
        ],
        "objectives": [
            "dynamic business development professional focused on driving revenue and market expansion",
            "strategic thinker with proven ability to identify and convert new business opportunities",
            "seeking to leverage BD expertise to scale company partnerships and customer base",
            "results-oriented BD manager with strong track record in B2B deal closing",
        ],
    },

    "SALES": {
        "skills": [
            "sales strategy", "cold calling", "lead conversion",
            "territory management", "retail sales", "FMCG sales",
            "client relationship management", "upselling", "cross-selling",
            "sales forecasting", "target achievement", "product demonstration",
            "dealer management", "sales reporting", "negotiation",
        ],
        "tools": [
            "Salesforce", "Zoho CRM", "MS Excel", "POS systems",
            "LinkedIn", "sales analytics dashboards",
        ],
        "certs": [
            "MBA Sales and Marketing", "BSc Commerce",
            "Diploma in Sales Management", "Certified Sales Professional",
        ],
        "roles": [
            "sales executive", "sales manager", "account manager",
            "business development executive", "retail sales associate",
            "territory sales representative", "inside sales specialist",
        ],
        "objectives": [
            "target-driven sales professional with consistent record of exceeding revenue goals",
            "passionate sales executive skilled in building customer relationships and closing deals",
            "seeking a senior sales role to drive business growth through strategic territory management",
            "dynamic sales specialist focused on delivering value to clients and maximizing revenue",
        ],
    },

    "PUBLIC-RELATIONS": {
        "skills": [
            "media relations", "press release writing", "crisis communications",
            "brand storytelling", "event management", "stakeholder engagement",
            "public affairs", "corporate communications", "social media PR",
            "reputation management", "speech writing", "content strategy",
            "journalist relations", "government relations",
        ],
        "tools": [
            "Cision", "Meltwater", "PR Newswire", "Hootsuite",
            "Google Analytics", "media monitoring tools",
        ],
        "certs": [
            "BA Mass Communication", "MA Public Relations",
            "PRSA APR Certification", "Diploma in Communications",
        ],
        "roles": [
            "PR executive", "communications manager", "media relations officer",
            "corporate communications specialist", "PR consultant",
            "brand communications manager",
        ],
        "objectives": [
            "strategic PR professional with expertise in media relations and brand reputation management",
            "seeking a senior communications role to drive compelling brand narratives",
            "experienced PR consultant with a network of media contacts across industry sectors",
            "passionate storyteller committed to building authentic public relations campaigns",
        ],
    },

    "BANKING": {
        "skills": [
            "retail banking", "credit analysis", "loan processing",
            "risk management", "KYC compliance", "AML procedures",
            "treasury operations", "relationship banking", "trade finance",
            "financial products", "branch operations", "wealth management",
            "mortgage lending", "Basel III", "credit underwriting",
        ],
        "tools": [
            "Finacle", "SAP Banking", "Bloomberg Terminal",
            "SWIFT", "core banking systems", "loan origination software",
        ],
        "certs": [
            "MBA Finance", "BCom", "Chartered Banker",
            "CAIIB", "JAIIB", "CFA Level 1",
        ],
        "roles": [
            "bank manager", "credit analyst", "relationship manager",
            "loan officer", "treasury analyst", "compliance officer",
            "branch operations manager", "wealth manager",
        ],
        "objectives": [
            "banking professional with strong expertise in credit analysis and retail banking operations",
            "seeking a relationship management role to deepen client engagement and grow the loan portfolio",
            "compliance-focused banking executive with deep knowledge of AML and KYC frameworks",
            "dedicated banker aiming to deliver superior financial solutions to individual and corporate clients",
        ],
    },
}

# ── Template câu ────────────────────────────────────────────────────────────────
OBJ_TEMPLATES = [
    "{obj} with {exp} years of experience in {s1} and {s2}.",
    "{obj}. Proven expertise in {s1}, {s2}, and {s3}.",
    "Motivated {role} with a passion for {s1}. {obj}.",
    "Dynamic {role} specializing in {s1} and {s2}. {obj}.",
]

SKILL_TEMPLATES = [
    "Proficient in {s1}, {s2}, and {s3}. Experienced with {t1} and {t2}.",
    "Core competencies include {s1}, {s2}, {s3}, and {s4}.",
    "Strong background in {s1} with hands-on experience using {t1} and {t2}.",
    "Key skills: {s1}, {s2}, {s3}. Familiar with {t1}.",
    "Expertise spans {s1}, {s2}, and {s3}, supported by proficiency in {t1}.",
]

EXP_TEMPLATES = [
    "Worked as {role} for {exp} years, handling {s1} and {s2} responsibilities.",
    "Served as {role} at a leading organization, managing {s1} and overseeing {s2}.",
    "{exp}+ years as {role} with a strong focus on {s1}, {s2}, and team collaboration.",
    "Successfully delivered {s1} projects as {role}. Collaborated cross-functionally on {s2}.",
]

EDU_TEMPLATES = [
    "Holds a {cert} with specialization in relevant areas.",
    "Educational background: {cert}. Continuously updating skills in the field.",
    "Completed {cert}. Strong academic foundation complemented by practical experience.",
    "{cert} graduate with distinction. Committed to ongoing professional development.",
]

CLOSING_TEMPLATES = [
    "Seeking to contribute meaningfully to a forward-thinking organization.",
    "Eager to leverage skills to drive organizational growth and excellence.",
    "Looking forward to taking on new challenges and growing professionally.",
    "Committed to continuous learning and delivering high-impact results.",
]


def _pick(lst, k=1, exclude=None):
    pool = [x for x in lst if x != exclude]
    return random.sample(pool, min(k, len(pool)))


def generate_resume(category, pool):
    skills  = pool["skills"]
    tools   = pool["tools"]
    certs   = pool["certs"]
    roles   = pool["roles"]
    objs    = pool["objectives"]

    role = random.choice(roles)
    cert = random.choice(certs)
    obj  = random.choice(objs)
    exp  = random.randint(1, 12)

    s_sample = _pick(skills, 6)
    t_sample = _pick(tools,  3)

    # Objective sentence
    sent_obj = random.choice(OBJ_TEMPLATES).format(
        obj=obj, exp=exp, role=role,
        s1=s_sample[0], s2=s_sample[1], s3=s_sample[2]
    )

    # Skills sentence
    sent_skill = random.choice(SKILL_TEMPLATES).format(
        s1=s_sample[0], s2=s_sample[1], s3=s_sample[2], s4=s_sample[3],
        t1=t_sample[0], t2=t_sample[1] if len(t_sample) > 1 else t_sample[0]
    )

    # Experience sentence
    sent_exp = random.choice(EXP_TEMPLATES).format(
        role=role, exp=exp, s1=s_sample[3], s2=s_sample[4]
    )

    # Education sentence
    sent_edu = random.choice(EDU_TEMPLATES).format(cert=cert)

    # Extra skills mention
    extra = _pick(skills, 3, exclude=None)
    sent_extra = f"Additional expertise in {extra[0]}, {extra[1]}, and {extra[2]}."

    # Closing
    sent_close = random.choice(CLOSING_TEMPLATES)

    # Randomize paragraph order slightly
    parts = [sent_obj, sent_skill, sent_exp, sent_edu, sent_extra, sent_close]
    random.shuffle(parts[2:4])   # swap exp/edu sometimes
    return " ".join(parts)


def generate_synthetic_dataset(target_per_class=200, output_path="Data/Raw/synthetic_resumes.csv"):
    """Sinh synthetic CV cho các class dưới target_per_class mẫu."""
    existing = pd.read_csv("Data/Processed/cleaned_resume.csv")
    dist     = existing["Category"].value_counts()

    records = []
    total_generated = 0

    print(f"Sinh du lieu tong hop (target = {target_per_class} mau/class)...\n")

    for category, pool in POOLS.items():
        current = dist.get(category, 0)
        needed  = max(0, target_per_class - current)

        if needed == 0:
            print(f"  {category:<25} da du {current} mau — bo qua")
            continue

        print(f"  {category:<25} {current:>4} mau → sinh them {needed} mau")

        for i in range(needed):
            text = generate_resume(category, pool)
            records.append({
                "ID":          f"SYN_{category[:3]}_{i:04d}",
                "Resume_str":  text,
                "Category":    category,
            })
            total_generated += 1

    df_syn = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_syn.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nTong so mau da sinh: {total_generated}")
    print(f"Da luu: {output_path}")
    print("\nPhan bo theo class:")
    for cat, cnt in df_syn["Category"].value_counts().sort_index().items():
        print(f"  {cat:<25} {cnt}")

    return df_syn


if __name__ == "__main__":
    generate_synthetic_dataset(target_per_class=200)
