"""Seed 5 realistic sample calls through the REAL scoring pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app import models, scoring
from app.ai.providers import get_provider

CALL_1 = """Agent: Hi Dana, thanks for taking the time. How is your week going so far?
Prospect: Busy but good, thanks. I only have about twenty minutes.
Agent: Perfect, I will keep this tight. What prompted you to take this call today?
Prospect: We are drowning in support tickets and our current tool cannot keep up.
Agent: That sounds painful. How many tickets is your team handling each week?
Prospect: Around four hundred, with three agents.
Agent: Got it. And how are you measuring resolution time right now?
Prospect: Honestly we are not, that is part of the problem.
Agent: What would a great outcome look like for you in the next quarter?
Prospect: Cut first response time under an hour and stop missing SLAs.
Agent: Makes sense. Who else would weigh in on a decision like this?
Prospect: Me and our COO, final sign off is hers.
Agent: Good to know. What kind of budget range have you set aside for tooling like this?
Prospect: We have about two thousand a month approved, but the price needs to make sense.
Agent: Understood. When are you hoping to have something live?
Prospect: End of next month ideally.
Agent: That is doable. One concern I hear a lot is pricing, is the per-seat cost a worry for you?
Prospect: A bit, yes. Your pricing page looks expensive compared to what we pay now.
Agent: Fair point. What if we start with a pilot on your three agents and only scale when the ROI is proven? That keeps the first invoice well under your budget.
Prospect: That could work. What does the pilot include?
Agent: Full onboarding, SLA reporting, and a dedicated success manager for thirty days. If response times do not improve, you walk away with the data.
Prospect: I like that. What is the next step?
Agent: I will send a proposal with the pilot terms today, and we can schedule a technical review with your COO on Thursday. Does that work?
Prospect: Thursday works. Send the proposal over.
Agent: Will do. Thanks Dana, talk Thursday."""

CALL_2 = """Agent: Hi Tom, appreciate you hopping on. What is top of mind for your support team right now?
Prospect: We need better reporting. Our manager wants real numbers by end of quarter.
Agent: Makes sense. How are you tracking performance today?
Prospect: Spreadsheets, honestly. It is a mess.
Agent: Painful. Walk me through what the spreadsheet workflow looks like.
Prospect: Each agent logs their tickets, I merge everything on Fridays, and it takes half my day.
Agent: That is a real time sink. What is driving the timeline, is there a renewal or a board meeting?
Prospect: Board meeting in six weeks. We need the numbers by then.
Agent: Understood. What budget have you earmarked for this?
Prospect: Around fifteen hundred a month.
Agent: That fits our team plan. Who else would need to weigh in before we move forward?
Prospect: Just me for now, I run the department.
Agent: Great. One thing teams ask about is the learning curve. Is a two-week rollout fast enough for your board deadline?
Prospect: That should be fine. My one worry is whether the team will actually use it.
Agent: Fair concern. We include live training for all eight agents, and adoption reports show you exactly who is logging in, so nothing slips.
Prospect: Okay, that helps. What happens next?
Agent: I will send a proposal with the team plan and the training schedule, then we can lock in a follow up next Wednesday to review it.
Prospect: Wednesday works. Send it over."""

CALL_3 = """Agent: Hi Lena, thanks for taking my call. Let me tell you a bit about what we do before we dive in. We help support teams cut response times with AI triage, smart routing, and real-time coaching. Does any of that sound relevant to your team?
Prospect: Some of it, yes. We are mostly struggling with ticket volume.
Agent: Volume is exactly what we solve. Our AI drafts replies, so agents handle twice the tickets. What is your current ticket volume?
Prospect: About three hundred a week, and it keeps growing.
Agent: That is a lot for a small team. How are your agents coping with the growth?
Prospect: They are not, honestly. Two people quit last month.
Agent: That is tough. Getting ahead of this quickly matters then. When would you want something live?
Prospect: As soon as possible, really. This quarter if we can.
Agent: Doable. What budget range are we working with?
Prospect: We have not really set one, but the price needs to be reasonable. Your enterprise tier looks expensive.
Agent: Fair point on price. What if we start on the standard tier, which is half the cost, and only move up when the ROI is proven? That keeps the first invoice reasonable.
Prospect: That sounds more reasonable. But an annual contract is a concern. We got locked into a bad one before.
Agent: Our terms are flexible, and we can go through them together on a call whenever you are ready.
Prospect: Okay. What are the next steps?
Agent: I will send a proposal with the standard tier pricing this afternoon, and we can schedule a follow up for Thursday to walk through it."""

CALL_4 = """Agent: Hi Rob, thanks for picking up. I will keep this brief and just walk you through what our platform does, because I think it is a great fit. We do AI triage, smart routing, real-time coaching, and the analytics suite is best in class. Sound good?
Prospect: I guess. We are pretty busy.
Agent: Totally get it. The triage alone saves teams ten hours a week. The setup takes a day, the dashboard is intuitive, and the ROI is usually visible in the first month. When are you looking to make a change?
Prospect: Not really looking. Someone just booked this for me.
Agent: That happens a lot, and most of those calls turn into great partnerships. The coaching feature listens to calls and gives agents live suggestions, which lifts CSAT fast. Let me show you the numbers.
Prospect: We have no budget for new tools this year.
Agent: A lot of teams say that at first. The platform usually pays for itself within two months, and we can start small on a monthly plan so there is no big commitment.
Prospect: Monthly is still money. I am not convinced this is a priority.
Agent: I hear that. Priorities shift, and when support volume spikes again you will want this in place. I will send over some materials you can look at when the time is right.
Prospect: Fine."""

CALL_5 = """Agent: Hi Sam, is now a good time for a quick call?
Prospect: Not really, I have two minutes.
Agent: Perfect, I will be fast. Our platform uses AI to transform support operations end to end. Triage, routing, coaching, analytics, all in one place. Teams love it.
Prospect: We already have a tool.
Agent: Most teams do before they switch. Ours is faster to deploy and the AI is genuinely better. I can send you a comparison deck.
Prospect: I do not need a deck. We are fine, and honestly your pricing is too much for what it is.
Agent: Totally fair. A lot of people say that until they see the automation in action. What would it take for you to look at a demo?
Prospect: Nothing right now. I have to go.
Agent: Understood. Thanks for your time, Sam."""

SEED_CALLS = [
    {
        "contact_name": "Dana Whitfield",
        "contact_phone": "+1 415 555 0132",
        "agent": "Maya Santos",
        "days_ago": 1,
        "duration_sec": 1180,
        "transcript": CALL_1,
    },
    {
        "contact_name": "Tom Beck",
        "contact_phone": "+1 312 555 0177",
        "agent": "Jake Rivera",
        "days_ago": 2,
        "duration_sec": 960,
        "transcript": CALL_2,
    },
    {
        "contact_name": "Lena Ortiz",
        "contact_phone": "+1 718 555 0119",
        "agent": "Maya Santos",
        "days_ago": 4,
        "duration_sec": 840,
        "transcript": CALL_3,
    },
    {
        "contact_name": "Rob Hale",
        "contact_phone": "+1 206 555 0148",
        "agent": "Priya Nair",
        "days_ago": 6,
        "duration_sec": 620,
        "transcript": CALL_4,
    },
    {
        "contact_name": "Sam Delgado",
        "contact_phone": "+1 617 555 0163",
        "agent": "Jake Rivera",
        "days_ago": 8,
        "duration_sec": 300,
        "transcript": CALL_5,
    },
]


def seed_product(db) -> dict:
    existing = {row[0] for row in db.query(models.Call.contact_name).all()}
    if any(c["contact_name"] in existing for c in SEED_CALLS):
        return {"calls": 0}
    provider = get_provider()
    base = datetime(2026, 9, 27, 9, 0, 0, tzinfo=timezone.utc)
    count = 0
    for c in SEED_CALLS:
        turns = scoring.parse_turns(c["transcript"])
        result = scoring.score_call(turns)
        summary = provider.summarize(scoring.plain_text(turns), max_sentences=4)
        db.add(
            models.Call(
                contact_name=c["contact_name"],
                contact_phone=c["contact_phone"],
                agent=c["agent"],
                started_at=base - timedelta(days=c["days_ago"]),
                duration_sec=c["duration_sec"],
                transcript=c["transcript"],
                score=result["score"],
                outcome=result["outcome"],
                talk_ratio=result["talk_ratio"],
                questions_asked=result["questions_asked"],
                summary=summary,
                highlights_json=json.dumps(result["highlights"]),
                coaching_json=json.dumps(result["coaching"]),
            )
        )
        count += 1
    db.flush()
    return {"calls": count}
