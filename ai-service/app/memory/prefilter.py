"""Pre-filter: skip the extraction call for turns that cannot hold anything worth remembering (plan §07).

Rule based, no model call. False negatives are cheap (the turn stays in the summary);
false positives only cost one lightweight extraction call.
"""

from __future__ import annotations

import re

_TRIVIAL = re.compile(r"^[\sㅋㅎㅠㅜ.!?~^]*$")
_SHORT_ACKS = {
    "응", "어", "웅", "ㅇㅇ", "ㅇㅋ", "오케이", "그래", "그치", "맞아", "아니", "헐", "대박", "진짜", "굿", "넹", "네", "예",
    "좋아", "알겠어", "고마워", "ㄱㅅ", "ㄴㄴ", "뭐해", "안녕", "잘자", "좋은 아침",
}
_SIGNALS = re.compile(
    r"(좋아해|싫어해|좋아하|싫어하|알레르기|생일|면접|시험|약속|여행|이사|회사|학교|직장|취업|합격|불합격|"
    r"헤어|사귀|가족|엄마|아빠|동생|언니|오빠|형|누나|고민|힘들|우울|걱정|미안|사과|화났|싸웠|"
    r"내일|모레|다음 ?주|담주|이번 ?주|주말|월요일|화요일|수요일|목요일|금요일|토요일|일요일|\d+월|\d+일|"
    r"(?:에|\s)\s*살(?:아|고|았|거든)|키워(?!드)|키우|키운|기억해|잊지 ?마|이제부터|사실은|비밀)"
)


def worth_extracting(user_text: str | None, assistant_text: str, min_chars: int = 12) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False
    if _TRIVIAL.match(text):
        return False
    normalized = re.sub(r"[\s.!?~ㅋㅎ]+$", "", text)
    if normalized in _SHORT_ACKS:
        return False
    if _SIGNALS.search(text):
        return True
    # Longer free-form statements may still carry facts; very short ones rarely do.
    return len(normalized) >= min_chars
