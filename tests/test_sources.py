"""Normalising Remotive API rows into JobIn."""
from jobpilot.ingest.sources.remotive import infer_level, normalise

RAW = {
    "id": 123456,
    "url": "https://remotive.com/remote-jobs/software-dev/backend-engineer-123456",
    "title": "Senior Backend Engineer (Python)",
    "company_name": "Acme Remote",
    "category": "Software Development",
    "tags": ["python", "django", "aws", "postgres"],
    "job_type": "full_time",
    "publication_date": "2026-11-18T10:00:00",
    "candidate_required_location": "Worldwide",
    "salary": "$90k - $120k",
    "description": "<p>We need <b>5+ years</b> building APIs with Python &amp; Django.</p><ul><li>Docker</li></ul>",
}


def test_normalise_remotive_job():
    job = normalise(RAW)
    assert job is not None
    assert job.source == "remotive" and job.external_id == "123456" and job.remote
    assert job.level == "senior" and job.experience_min == 5
    assert {"python", "django", "aws", "postgresql", "docker"} <= set(job.skills)
    assert job.description.startswith("Salary: $90k - $120k")
    assert "<p>" not in job.description and job.posted_at.tzinfo is not None


def test_bad_rows_are_skipped():
    assert normalise({"id": 1, "title": "", "company_name": "x"}) is None


def test_infer_level():
    assert infer_level("Software Engineering Intern", "")[0] == "intern"
    assert infer_level("Junior Developer", "")[0] == "junior"
    assert infer_level("Backend Developer", "requires 2 years of experience")[:2] == ("junior", 2.0)
    assert infer_level("Backend Developer", "")[0] == "mid"
