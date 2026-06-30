from __future__ import annotations


BULLET_LINE_RE = r"- [^\n]{1,300}"
NARRATIVE_FOCUS_NAME_RE = r"\S(?:[^\n;]{0,63}\S)?"
NARRATIVE_FOCUSES_RE = NARRATIVE_FOCUS_NAME_RE + r"(?:; " + NARRATIVE_FOCUS_NAME_RE + r"){0,6}"
EMBEDDING_SPACE_RE = (
    r"action: [0-9]{1,2}, dialog: [0-9]{1,2}, world_building: [0-9]{1,2}, exposition: [0-9]{1,2}, "
    r"romantic: [0-9]{1,2}, erotic: [0-9]{1,2}, pacing: [0-9]{1,2}"
)
SCENE_EMBEDDING_SPACE_RE = (
    r"action: [0-9]{1,2}, dialog: [0-9]{1,2}, world_building: [0-9]{1,2}, exposition: [0-9]{1,2}, "
    r"romantic: [0-9]{1,2}, erotic: [0-9]{1,2}, pacing: [0-9]{1,2}"
)
ARC_HEADER_RE = r"### Arc [1-9][0-9]*: [^\n]{1,120}\n"
SHORT_STORY_ARC_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:{BULLET_LINE_RE}\n){{1,20}}\n?){{1,12}}"
CHARACTER_ARCHETYPE_SECTION_RE = rf"(?:### [^\n]{{1,120}}\n(?:{BULLET_LINE_RE}\n){{1,10}}\n?){{1,32}}"
WRITING_STYLE_SECTION_RE = rf"(?:{BULLET_LINE_RE}\n){{0,39}}{BULLET_LINE_RE}\n?"


def _build_non_header_text_line_regex(max_total_characters: int, max_leading_spaces: int) -> str:
    variants: list[str] = []
    for leading_space_count in range(max_leading_spaces + 1):
        remaining_characters = max_total_characters - leading_space_count - 1
        variants.append(" " * leading_space_count + rf"[^\s#\n][^\n]{{0,{remaining_characters}}}")
    return "(?:" + "|".join(variants) + ")"


NON_HEADER_TEXT_LINE_RE = _build_non_header_text_line_regex(max_total_characters=800, max_leading_spaces=12)
BOOK_TITLE_LINE_RE = _build_non_header_text_line_regex(max_total_characters=120, max_leading_spaces=12)
BOOK_HIGHLIGHT_BODY_RE = rf"(?:{NON_HEADER_TEXT_LINE_RE}\n){{0,9}}{NON_HEADER_TEXT_LINE_RE}"
BOOK_ARCHETYPE_BODY_RE = rf"(?:{NON_HEADER_TEXT_LINE_RE}\n){{0,11}}{NON_HEADER_TEXT_LINE_RE}"
BOOK_PREVIEW_TEXT_WORD_RE = (
    r"(?:[A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ][A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ&'\"`´/().,:;!?+—–“”‘’…-]*"
    r"|[&'\"`´/().,:;!?+—–“”‘’…-][A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ]"
    r"[A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ&'\"`´/().,:;!?+—–“”‘’…-]*"
    r"|&|[—–“”‘’…])"
)
BOOK_PREVIEW_TAG_WORD_RE = (
    r"(?:[A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ][A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ&'\"`´/().:+—–“”‘’…-]*"
    r"|[&'\"`´/().:+—–“”‘’…-][A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ]"
    r"[A-Za-z0-9À-ÖØ-öø-ÿĀ-ſ&'\"`´/().:+—–“”‘’…-]*"
    r"|&|[—–“”‘’…])"
)
BOOK_PREVIEW_TEXT_BODY_RE = (
    r"("
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"(?:[^\S\n]+"
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"){54,139})"
)
BOOK_PREVIEW_TITLE_BODY_RE = (
    r"("
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"(?:[^\S\n]+"
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"){0,7})"
)
BOOK_PREVIEW_TAGS_BODY_RE = (
    r"((?:- "
    + BOOK_PREVIEW_TAG_WORD_RE
    + r"(?:[^\S\n]+"
    + BOOK_PREVIEW_TAG_WORD_RE
    + r"){0,7}\n){7})"
)
BOOK_PREVIEW_ARCHETYPE_BODY_RE = (
    r"("
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"(?:[^\S\n]+"
    + BOOK_PREVIEW_TEXT_WORD_RE
    + r"){44,189})"
)
CHAPTER_SUMMARY_BULLETS_RE_24B = rf"(?:{BULLET_LINE_RE}\n){{0,31}}{BULLET_LINE_RE}\n?"
SCENE_SUMMARY_BULLETS_RE = rf"(?:{BULLET_LINE_RE}\n){{0,19}}{BULLET_LINE_RE}\n?"
CHAPTER_PLAN_BLOCK_RE_24B = (
    r"### [^\n]{1,120}\n"
    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
    r"\*\*Embedding Space:\*\* "
    + EMBEDDING_SPACE_RE
    + r"\n"
    r"\*\*Narrative Focuses:\*\* "
    + NARRATIVE_FOCUSES_RE
    + r"\n"
    r"\*\*Chapter Summary:\*\*\n"
    + CHAPTER_SUMMARY_BULLETS_RE_24B
)
SCENE_BREAKDOWN_BLOCK_RE = (
    r"#### Scene (?:[1-9]|1[0-2]): [^\n]{1,120}\n"
    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
    r"\*\*Embedding Space:\*\* "
    + SCENE_EMBEDDING_SPACE_RE
    + r"\n"
    r"\*\*Narrative Focus:\*\* "
    + NARRATIVE_FOCUSES_RE
    + r"\n"
    r"\*\*Narrative Perspective:\*\* [^\n]{1,200}\n"
    r"\*\*Scene Summary:\*\*\n"
    + SCENE_SUMMARY_BULLETS_RE
)

BOOK_PREVIEW_REGEX = (
    r"\A## Book Highlight\n"
    + BOOK_PREVIEW_TEXT_BODY_RE
    + r"\n\n## Book Title\n"
    + BOOK_PREVIEW_TITLE_BODY_RE
    + r"\n\n## Book Tags\n"
    + BOOK_PREVIEW_TAGS_BODY_RE
    + r"\n## Book Archetype\n"
    + BOOK_PREVIEW_ARCHETYPE_BODY_RE
    + r"\z"
)
BOOK_PLAN_PREVIEW_REGEX_24B = BOOK_PREVIEW_REGEX.removesuffix(r"\z") + r"\n\n## World Rules\n\z"
BOOK_PLAN_REMAINDER_REGEX_24B = (
    rf"^(?:{BULLET_LINE_RE}\n){{1,90}}"
    r"\n## Story Arcs\n"
    + SHORT_STORY_ARC_SECTION_RE
    + r"## Character Archetypes\n"
    + CHARACTER_ARCHETYPE_SECTION_RE
    + r"## Writing Style\n"
    + WRITING_STYLE_SECTION_RE
    + r"$"
)
BOOK_PLAN_REGEX_24B = (
    BOOK_PLAN_PREVIEW_REGEX_24B.removesuffix(r"\z")
    + BOOK_PLAN_REMAINDER_REGEX_24B.removeprefix("^")
)
FIRST_CHAPTER_PLAN_REGEX_24B = r"^(?:" + CHAPTER_PLAN_BLOCK_RE_24B + r"\n?){1,2}$"
SCENE_BREAKDOWN_REGEX = r"^### [^\n]{1,120}\n(?:" + SCENE_BREAKDOWN_BLOCK_RE + r"\n?){1,12}$"
