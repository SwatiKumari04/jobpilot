"""Deterministic synthetic job generator.

Why synthetic data? Every job has a known `category`, so we get free
ground-truth relevance labels for the retrieval eval (evals/retrieval_eval.py).
Company names are invented. These listings are labelled "demo" in the UI.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from ..pipeline import JobIn

CATEGORIES: dict[str, dict] = {
    "backend": {
        "titles": ["Backend Engineer", "Software Engineer - Backend", "Python Developer", "Java Backend Developer", "API Engineer"],
        "skills": ["python", "fastapi", "django", "sql", "postgresql", "redis", "docker", "rest api", "java", "spring boot", "microservices", "kafka", "aws"],
        "duties": ["design and build REST APIs and server-side services", "own database schemas and write efficient SQL queries",
                   "improve reliability and latency of production services", "work on authentication, payments and background workers",
                   "scale services that handle millions of requests a day"],
    },
    "frontend": {
        "titles": ["Frontend Engineer", "React Developer", "UI Engineer", "Web Developer"],
        "skills": ["javascript", "typescript", "react", "next.js", "html", "css", "figma", "testing"],
        "duties": ["build responsive user interfaces in React", "turn Figma designs into pixel-perfect pages",
                   "improve page load performance and accessibility", "write reusable UI components and a design system",
                   "collaborate with designers on interactions and animations"],
    },
    "fullstack": {
        "titles": ["Full Stack Developer", "Full Stack Engineer (MERN)", "Software Engineer - Full Stack"],
        "skills": ["javascript", "react", "node.js", "express", "mongodb", "typescript", "sql", "docker", "rest api"],
        "duties": ["ship features end to end, from React screens to Node.js APIs", "model data in MongoDB and SQL",
                   "own small products from idea to production", "write clean code across the stack and review pull requests"],
    },
    "ml_ai": {
        "titles": ["Machine Learning Engineer", "AI Engineer", "LLM Engineer", "Applied Scientist", "NLP Engineer"],
        "skills": ["python", "pytorch", "machine learning", "llm", "rag", "embeddings", "vector database", "nlp",
                   "prompt engineering", "agents", "scikit-learn", "deep learning"],
        "duties": ["build LLM-powered features with retrieval and tool calling", "train and evaluate models on text data",
                   "ship recommendation and ranking models to production", "set up evaluation pipelines for model quality",
                   "experiment with prompts, embeddings and fine-tuning"],
    },
    "data": {
        "titles": ["Data Analyst", "Business Analyst", "Data Engineer", "Analytics Engineer"],
        "skills": ["sql", "python", "pandas", "excel", "power bi", "tableau", "statistics", "spark", "airflow", "data analysis"],
        "duties": ["build dashboards that leadership uses every week", "write SQL to answer product questions",
                   "build reliable data pipelines", "analyse A/B tests and present the findings", "clean and model messy datasets"],
    },
    "devops": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
        "skills": ["docker", "kubernetes", "aws", "terraform", "linux", "ci/cd", "gcp", "azure", "python"],
        "duties": ["run and monitor our cloud infrastructure", "automate deployments with CI/CD pipelines",
                   "manage infrastructure as code with Terraform", "handle on-call incidents and improve uptime",
                   "cut cloud costs without hurting performance"],
    },
    "mobile": {
        "titles": ["Android Developer", "iOS Developer", "Flutter Developer", "Mobile Engineer"],
        "skills": ["kotlin", "android", "swift", "ios", "flutter", "react native", "rest api", "git"],
        "duties": ["build smooth Android and iOS apps", "ship features to millions of smartphone users",
                   "optimise app startup time and battery usage", "integrate push notifications and offline sync"],
    },
    "qa": {
        "titles": ["QA Engineer", "SDET", "Test Automation Engineer"],
        "skills": ["selenium", "testing", "python", "java", "ci/cd", "rest api"],
        "duties": ["write automated tests for web apps and APIs", "own the regression suite in CI",
                   "track down flaky tests and hard-to-reproduce defects", "define test plans with product managers"],
    },
    "embedded": {
        "titles": ["Embedded Software Engineer", "Firmware Engineer", "Embedded Systems Engineer"],
        "skills": ["c", "c++", "embedded", "linux", "python", "git"],
        "duties": ["write firmware for microcontrollers", "work close to hardware: sensors, motors and power electronics",
                   "debug boards with oscilloscopes and logic analysers", "write device drivers for embedded Linux"],
    },
}

LEVELS = [  # (level, exp_min, exp_max, salary_lo, salary_hi, weight)
    ("intern", 0, 0.5, 3, 12, 0.22),
    ("fresher", 0, 1, 5, 16, 0.24),
    ("junior", 1, 3, 7, 22, 0.24),
    ("mid", 3, 6, 15, 40, 0.18),
    ("senior", 5, 10, 30, 70, 0.12),
]
CITIES = ["Bengaluru", "Gurugram", "Hyderabad", "Pune", "Mumbai", "Noida", "Chennai", "Kolkata"]
PREFIX = ["Nimbus", "Kestrel", "Quanta", "Orbit", "Saffron", "Indigo", "Lattice", "Vertex", "Monsoon", "Banyan",
          "Peacock", "Zenith", "Cobalt", "Tandem", "Ember", "Pixel", "Harbor", "Summit", "Kite", "Lotus"]
SUFFIX = ["Labs", "Tech", "AI", "Systems", "Pay", "Health", "Logistics", "Analytics", "Cloud", "Robotics"]
INDUSTRY = ["fintech", "healthtech", "edtech", "logistics", "e-commerce", "SaaS", "gaming", "climate-tech", "HR-tech"]
PERKS = ["Flexible hours and a learning budget.", "Health insurance for you and your family.",
         "ESOPs for early team members.", "Mentorship from senior engineers.", "Free meals and a hybrid setup."]
GHOST_TEXT = ("We are always hiring talented people for future opportunities. "
              "Send your profile to join our talent pool.")


def generate(n: int = 300, seed: int = 42, now: datetime | None = None) -> list[JobIn]:
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    companies = [f"{p} {s}" for p in PREFIX for s in SUFFIX]
    rng.shuffle(companies)
    jobs: list[JobIn] = []
    cats = list(CATEGORIES)
    level_weights = [lv[-1] for lv in LEVELS]
    for i in range(n):
        cat = cats[i % len(cats)]
        spec = CATEGORIES[cat]
        level, emin, emax, slo, shi, _ = rng.choices(LEVELS, weights=level_weights)[0]
        base_title = rng.choice(spec["titles"])
        title = f"{base_title} Intern" if level == "intern" else (f"Senior {base_title}" if level == "senior" else base_title)
        company = companies[i % len(companies)]
        remote = rng.random() < 0.25
        location = "Remote (India)" if remote else rng.choice(CITIES)
        skills = rng.sample(spec["skills"], k=min(len(spec["skills"]), rng.randint(4, 6)))
        duties = rng.sample(spec["duties"], k=3)
        ghost = rng.random() < 0.08
        if ghost:
            posted = now - timedelta(days=rng.randint(65, 120))
            description = f"{company} - {title}. {GHOST_TEXT} Skills: {', '.join(skills[:3])}."
            smin = smax = None
        else:
            posted = now - timedelta(days=rng.randint(0, 40), hours=rng.randint(0, 23))
            lo = rng.randint(slo, (slo + shi) // 2)
            smin, smax = float(lo), float(min(shi, lo + rng.randint(3, 12)))
            description = (
                f"{company} is a {rng.choice(INDUSTRY)} startup. As a {title} you will {duties[0]}, {duties[1]} "
                f"and {duties[2]}. Must have: {', '.join(skills[:3])}. Nice to have: {', '.join(skills[3:]) or 'curiosity'}. "
                f"Experience: {emin:g}-{emax:g} years. Location: {location}. {rng.choice(PERKS)}"
            )
            if rng.random() < 0.15:
                smin = smax = None  # many real listings hide the salary
        jobs.append(JobIn(
            source="synthetic", external_id=f"syn-{i}", title=title, company=company, location=location,
            remote=remote, category=cat, level=level, experience_min=emin, experience_max=emax,
            salary_min_lpa=smin, salary_max_lpa=smax, skills=skills, description=description,
            url="", posted_at=posted,
        ))
    # ~5% cross-posts: the same job seen on another board (tests dedupe)
    for k, original in enumerate(rng.sample(jobs, k=max(1, n // 20))):
        jobs.append(original.model_copy(update={"external_id": f"syn-board2-{k}"}))
    return jobs
