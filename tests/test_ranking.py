"""Reciprocal Rank Fusion and personalised scoring."""
from datetime import datetime, timedelta, timezone

from jobpilot.search.ranking import experience_fit, location_fit, rrf_fuse, score_job, skill_overlap

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def test_rrf_rewards_agreement():
    scores = rrf_fuse([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=scores.get) == 1     # ranks 1 and 2
    assert scores[3] > scores[2]                # in both lists beats in one
    assert 4 in scores


def test_skill_overlap():
    score, matched, missing = skill_overlap(["python", "sql", "docker", "kafka"], ["Python", "SQL"])
    assert score == 0.5 and matched == ["python", "sql"] and missing == ["docker", "kafka"]


def test_location_and_experience_fit():
    assert location_fit("Remote (India)", True, ["Pune"], True)[0] == 1.0
    assert location_fit("Bengaluru", False, ["bengaluru"], True)[0] == 1.0
    assert location_fit("Pune", False, ["Bengaluru"], True)[0] < 0.5
    assert experience_fit(0, 0, 1)[0] == 1.0
    assert experience_fit(0, 5, 10)[0] == 0.0


def _job(**kw):
    base = dict(id=1, title="t", company="c", location="Bengaluru", remote=False, skills=["python", "sql"],
                experience_min=0, experience_max=1, posted_at=NOW - timedelta(days=1), ghost_score=0.0,
                salary_max_lpa=10.0)
    base.update(kw)
    return base


def test_personalised_score_prefers_matching_job():
    profile = {"skills": ["python", "sql"], "years_experience": 0, "locations": ["Bengaluru"], "remote_ok": True}
    good, reasons, _ = score_job(_job(), profile, rel=0.8, now=NOW)
    bad, _, missing = score_job(_job(skills=["java", "kotlin"], location="Chennai", experience_min=5,
                                     experience_max=10), profile, rel=0.8, now=NOW)
    assert good > bad
    assert any("Matches 2/2" in r for r in reasons)
    assert missing == ["java", "kotlin"]


def test_ghost_listing_is_penalised():
    a, _, _ = score_job(_job(), None, rel=1.0, now=NOW)
    b, _, _ = score_job(_job(ghost_score=0.9), None, rel=1.0, now=NOW)
    assert a - b > 0.2


def test_salary_floor_penalty():
    profile = {"skills": [], "years_experience": 0, "locations": [], "remote_ok": True, "min_salary_lpa": 20}
    low, _, _ = score_job(_job(salary_max_lpa=8.0), profile, rel=1.0, now=NOW)
    ok, _, _ = score_job(_job(salary_max_lpa=25.0), profile, rel=1.0, now=NOW)
    assert ok > low
