from __future__ import annotations

import re
from typing import TypedDict


class PlannedChapterInfo(TypedDict):
    chapter_name: str
    chapter_word_count: int
    chapter_embedding_space: str


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
PLANNED_CHAPTER_SUMMARY_RE = rf"(?:{BULLET_LINE_RE}\n){{0,79}}{BULLET_LINE_RE}"
ARC_HEADER_RE = r"### Arc [1-9][0-9]*: [^\n]{1,120}\n"
ARC_CHAPTER_COUNT_RE = r"[1-9][0-9]?"
SHORT_STORY_ARC_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:{BULLET_LINE_RE}\n){{1,20}}\n?){{1,12}}"
MEDIUM_STORY_ARC_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:{BULLET_LINE_RE}\n){{1,40}}\n?){{1,12}}"
LONG_STORY_ARC_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:{BULLET_LINE_RE}\n){{1,80}}\n?){{1,12}}"
CHARACTER_ARCHETYPE_SECTION_RE = rf"(?:### [^\n]{{1,120}}\n(?:{BULLET_LINE_RE}\n){{1,10}}\n?){{1,32}}"
WRITING_STYLE_SECTION_RE = rf"(?:{BULLET_LINE_RE}\n){{0,39}}{BULLET_LINE_RE}\n?"
NUMBER_OF_CHAPTERS_SECTION_RE = rf"(?:{ARC_HEADER_RE}\* {ARC_CHAPTER_COUNT_RE}\n{{1,2}}){{1,12}}"
CHAPTER_NAMES_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:- [^\n]{{1,120}}\n){{1,99}}\n?){{1,12}}"
CHAPTERS_EMBEDDING_SPACE_SECTION_RE = (
    rf"(?:{ARC_HEADER_RE}(?:#### [^\n]{{1,120}}\n{EMBEDDING_SPACE_RE}\n){{1,99}}\n?){{1,12}}"
)
CHAPTERS_WORD_COUNT_SECTION_RE = rf"(?:{ARC_HEADER_RE}(?:#### [^\n]{{1,120}}\n[0-9]{{2,5}}\n?){{1,99}}\n?){{1,12}}"


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
SCENE_SUMMARY_BULLETS_RE = rf"(?:{BULLET_LINE_RE}\n){{0,19}}{BULLET_LINE_RE}\n?"
CHARACTER_BULLETS_RE = rf"(?:{BULLET_LINE_RE}\n){{0,19}}{BULLET_LINE_RE}\n?"
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
CHARACTER_BLOCK_RE = r"#### [^\n]{1,120}\n" + CHARACTER_BULLETS_RE

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
BOOK_PLAN_REGEX_14B = (
    r"^## World Rules\n"
    rf"(?:{BULLET_LINE_RE}\n){{1,90}}"
    r"\n## Short Story Arcs\n"
    + SHORT_STORY_ARC_SECTION_RE
    + r"\n?## Character Archetypes\n"
    + CHARACTER_ARCHETYPE_SECTION_RE
    + r"\n?## Writing Style\n"
    + WRITING_STYLE_SECTION_RE
    + r"\n?## Medium Story Arcs\n"
    + MEDIUM_STORY_ARC_SECTION_RE
    + r"\n?## Long Story Arcs\n"
    + LONG_STORY_ARC_SECTION_RE
    + r"\n?## Number of Chapters\n"
    + NUMBER_OF_CHAPTERS_SECTION_RE
    + r"\n?## Chapter Names\n"
    + CHAPTER_NAMES_SECTION_RE
    + r"\n?## Chapters Embedding Space\n"
    + CHAPTERS_EMBEDDING_SPACE_SECTION_RE
    + r"\n?## Chapters Word Count\n"
    + CHAPTERS_WORD_COUNT_SECTION_RE
    + r"$"
)
FIRST_CHAPTER_PLAN_REGEX_14B = (
    r"^### [^\n]{1,120}\n"
    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
    r"\*\*Embedding Space:\*\* "
    + EMBEDDING_SPACE_RE
    + r"\n"
    r"\*\*Narrative Focuses:\*\* "
    + NARRATIVE_FOCUSES_RE
    + r"\n"
    r"\*\*Chapter Summary:\*\*\n"
    + PLANNED_CHAPTER_SUMMARY_RE
    + r"\n\n### Scene Breakdown\n"
    + r"(?:" + SCENE_BREAKDOWN_BLOCK_RE + r"\n?){1,12}$"
)
SCENE_BREAKDOWN_REGEX = r"^### [^\n]{1,120}\n(?:" + SCENE_BREAKDOWN_BLOCK_RE + r"\n?){1,12}$"
BOOK_CHARACTERS_LIST_REGEX = (
    r"^### Main Characters\n(?:"
    + CHARACTER_BLOCK_RE
    + r"\n?){1,20}### Side Characters\n(?:"
    + CHARACTER_BLOCK_RE
    + r"\n?){1,20}$"
)


def _escape_regex_literal(literal_text: str) -> str:
    return re.escape(literal_text).replace(r"\ ", " ")


def _book_plan_section(book_plan_text: str, section_name: str) -> str:
    section_match = re.search(
        rf"(?ms)^## {re.escape(section_name)}\n(?P<body>.*?)(?=^## |\Z)",
        book_plan_text.strip(),
    )
    if section_match is None:
        raise ValueError(f"Missing book_plan section: {section_name}")
    return section_match.group("body").strip()


def extract_preloaded_chapters_from_book_plan(book_plan_text: str) -> list[PlannedChapterInfo]:
    arc_chapter_names = extract_arc_chapter_names_from_book_plan(book_plan_text)
    arc_embedding_spaces = _extract_arc_chapter_embedding_spaces_from_book_plan(book_plan_text)
    arc_word_counts = _extract_arc_chapter_word_counts_from_book_plan(book_plan_text)

    if [arc_title for arc_title, _ in arc_chapter_names] != [arc_title for arc_title, _ in arc_embedding_spaces]:
        raise ValueError("Chapter Names and Chapters Embedding Space arcs did not align in the 14b book_plan stage.")
    if [arc_title for arc_title, _ in arc_chapter_names] != [arc_title for arc_title, _ in arc_word_counts]:
        raise ValueError("Chapter Names and Chapters Word Count arcs did not align in the 14b book_plan stage.")

    planned_chapters: list[PlannedChapterInfo] = []
    for (arc_title, chapter_names), (_, chapter_embedding_spaces), (_, chapter_word_counts) in zip(
        arc_chapter_names,
        arc_embedding_spaces,
        arc_word_counts,
        strict=True,
    ):
        if len(chapter_names) != len(chapter_embedding_spaces):
            raise ValueError(
                "Chapter Names and Chapters Embedding Space counts did not align "
                f"for arc {arc_title!r} in the 14b book_plan stage."
            )
        if len(chapter_names) != len(chapter_word_counts):
            raise ValueError(
                "Chapter Names and Chapters Word Count counts did not align "
                f"for arc {arc_title!r} in the 14b book_plan stage."
            )

        for chapter_name, (embedding_chapter_name, chapter_embedding_space), (
            word_count_chapter_name,
            chapter_word_count,
        ) in zip(
            chapter_names,
            chapter_embedding_spaces,
            chapter_word_counts,
            strict=True,
        ):
            if chapter_name != embedding_chapter_name:
                raise ValueError(
                    "Chapter Names and Chapters Embedding Space did not align "
                    f"for chapter {chapter_name!r} in arc {arc_title!r}."
                )
            if chapter_name != word_count_chapter_name:
                raise ValueError(
                    "Chapter Names and Chapters Word Count did not align "
                    f"for chapter {chapter_name!r} in arc {arc_title!r}."
                )
            planned_chapters.append(
                {
                    "chapter_name": chapter_name,
                    "chapter_word_count": chapter_word_count,
                    "chapter_embedding_space": chapter_embedding_space,
                }
            )
    return planned_chapters


def _extract_arc_chapter_embedding_spaces_from_book_plan(book_plan_text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    section_text = _book_plan_section(book_plan_text, "Chapters Embedding Space")
    arc_embedding_spaces: list[tuple[str, list[tuple[str, str]]]] = []
    current_arc_title: str | None = None
    current_chapter_entries: list[tuple[str, str]] = []
    pending_chapter_name: str | None = None

    for line in section_text.splitlines():
        if line.startswith("### "):
            if pending_chapter_name is not None:
                raise ValueError(
                    f"Missing embedding-space value for chapter {pending_chapter_name!r} in the 14b book_plan stage."
                )
            if current_arc_title is not None:
                arc_embedding_spaces.append((current_arc_title, current_chapter_entries))
            current_arc_title = line.removeprefix("### ").strip()
            current_chapter_entries = []
            continue
        if line.startswith("#### "):
            if current_arc_title is None:
                raise ValueError("Encountered embedding-space chapter entries before an arc header in the 14b book_plan stage.")
            if pending_chapter_name is not None:
                raise ValueError(
                    f"Missing embedding-space value for chapter {pending_chapter_name!r} in the 14b book_plan stage."
                )
            pending_chapter_name = line.removeprefix("#### ").strip()
            continue
        if not line.strip():
            continue
        if pending_chapter_name is None:
            raise ValueError(f"Unsupported Chapters Embedding Space line in the 14b book_plan stage: {line!r}")
        if re.fullmatch(EMBEDDING_SPACE_RE, line.strip()) is None:
            raise ValueError(f"Invalid embedding-space value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
        current_chapter_entries.append((pending_chapter_name, line.strip()))
        pending_chapter_name = None

    if pending_chapter_name is not None:
        raise ValueError(f"Missing embedding-space value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
    if current_arc_title is not None:
        arc_embedding_spaces.append((current_arc_title, current_chapter_entries))
    if not arc_embedding_spaces:
        raise ValueError("Could not extract per-arc embedding spaces from the 14b book_plan stage.")
    return arc_embedding_spaces


def _extract_arc_chapter_word_counts_from_book_plan(book_plan_text: str) -> list[tuple[str, list[tuple[str, int]]]]:
    section_text = _book_plan_section(book_plan_text, "Chapters Word Count")
    arc_word_counts: list[tuple[str, list[tuple[str, int]]]] = []
    current_arc_title: str | None = None
    current_chapter_entries: list[tuple[str, int]] = []
    pending_chapter_name: str | None = None

    for line in section_text.splitlines():
        if line.startswith("### "):
            if pending_chapter_name is not None:
                raise ValueError(f"Missing word-count value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
            if current_arc_title is not None:
                arc_word_counts.append((current_arc_title, current_chapter_entries))
            current_arc_title = line.removeprefix("### ").strip()
            current_chapter_entries = []
            continue
        if line.startswith("#### "):
            if current_arc_title is None:
                raise ValueError("Encountered word-count chapter entries before an arc header in the 14b book_plan stage.")
            if pending_chapter_name is not None:
                raise ValueError(f"Missing word-count value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
            pending_chapter_name = line.removeprefix("#### ").strip()
            continue
        if not line.strip():
            continue
        if pending_chapter_name is None:
            raise ValueError(f"Unsupported Chapters Word Count line in the 14b book_plan stage: {line!r}")
        if re.fullmatch(r"[0-9]{2,5}", line.strip()) is None:
            raise ValueError(f"Invalid word-count value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
        current_chapter_entries.append((pending_chapter_name, int(line.strip())))
        pending_chapter_name = None

    if pending_chapter_name is not None:
        raise ValueError(f"Missing word-count value for chapter {pending_chapter_name!r} in the 14b book_plan stage.")
    if current_arc_title is not None:
        arc_word_counts.append((current_arc_title, current_chapter_entries))
    if not arc_word_counts:
        raise ValueError("Could not extract per-arc word counts from the 14b book_plan stage.")
    return arc_word_counts


def extract_arc_titles_from_book_plan(book_plan_text: str, section_name: str) -> list[str]:
    section_text = _book_plan_section(book_plan_text, section_name)
    arc_titles = [
        arc_title_match.group(1).strip()
        for arc_title_match in re.finditer(r"(?m)^### (Arc [1-9][0-9]*: .+?)\s*$", section_text)
    ]
    if not arc_titles:
        raise ValueError(f"Could not extract arc titles from the {section_name!r} section.")
    return arc_titles


def extract_arc_chapter_counts_from_book_plan(book_plan_text: str) -> list[tuple[str, int]]:
    section_text = _book_plan_section(book_plan_text, "Number of Chapters")
    arc_chapter_counts = [
        (count_match.group(1).strip(), int(count_match.group(2)))
        for count_match in re.finditer(
            rf"(?m)^### (Arc [1-9][0-9]*: [^\n]+?)\s*$\n\* ({ARC_CHAPTER_COUNT_RE})\s*$",
            section_text,
        )
    ]
    if not arc_chapter_counts:
        raise ValueError("Could not extract per-arc chapter counts from the 14b book_plan stage.")
    return arc_chapter_counts


def extract_arc_chapter_names_from_book_plan(book_plan_text: str) -> list[tuple[str, list[str]]]:
    section_text = _book_plan_section(book_plan_text, "Chapter Names")
    arc_chapter_names: list[tuple[str, list[str]]] = []
    current_arc_title: str | None = None
    current_chapter_names: list[str] = []

    for line in section_text.splitlines():
        if line.startswith("### "):
            if current_arc_title is not None:
                arc_chapter_names.append((current_arc_title, current_chapter_names))
            current_arc_title = line.removeprefix("### ").strip()
            current_chapter_names = []
            continue
        if line.startswith("- "):
            if current_arc_title is None:
                raise ValueError("Encountered chapter names before an arc header in the 14b book_plan stage.")
            current_chapter_names.append(line.removeprefix("- ").strip())
            continue
        if line.strip():
            raise ValueError(f"Unsupported Chapter Names line in the 14b book_plan stage: {line!r}")

    if current_arc_title is not None:
        arc_chapter_names.append((current_arc_title, current_chapter_names))
    if not arc_chapter_names:
        raise ValueError("Could not extract per-arc chapter names from the 14b book_plan stage.")
    return arc_chapter_names


def build_fourteen_b_book_plan_step_regexes() -> tuple[str, ...]:
    return (
        (
            r"^## World Rules\n"
            rf"(?:{BULLET_LINE_RE}\n){{1,90}}"
            r"\n## Short Story Arcs\n"
            + SHORT_STORY_ARC_SECTION_RE
            + r"\n?## Character Archetypes\n"
            + CHARACTER_ARCHETYPE_SECTION_RE
            + r"\n?## Writing Style\n"
            + WRITING_STYLE_SECTION_RE
            + r"$"
        ),
        r"^## Medium Story Arcs\n" + MEDIUM_STORY_ARC_SECTION_RE + r"$",
        r"^## Long Story Arcs\n" + LONG_STORY_ARC_SECTION_RE + r"$",
        r"^## Number of Chapters\n" + NUMBER_OF_CHAPTERS_SECTION_RE + r"$",
        r"^## Chapter Names\n" + CHAPTER_NAMES_SECTION_RE + r"$",
        r"^## Chapters Embedding Space\n" + CHAPTERS_EMBEDDING_SPACE_SECTION_RE + r"$",
        r"^## Chapters Word Count\n" + CHAPTERS_WORD_COUNT_SECTION_RE + r"$",
    )


def build_full_book_chapters_plan_regex(planned_chapters: list[PlannedChapterInfo]) -> str:
    if len(planned_chapters) < 2:
        raise ValueError("Dynamic full-book chapter-plan regex requires at least two planned chapters.")

    later_chapter_patterns = []
    for planned_chapter in planned_chapters[1:]:
        later_chapter_patterns.append(
            r"### " + _escape_regex_literal(planned_chapter["chapter_name"]) + r"\n"
            r"\*\*Word Count:\*\* " + _escape_regex_literal(str(planned_chapter["chapter_word_count"])) + r"\n"
            r"\*\*Embedding Space:\*\* " + _escape_regex_literal(planned_chapter["chapter_embedding_space"]) + r"\n"
            r"\*\*Narrative Focuses:\*\* " + NARRATIVE_FOCUSES_RE + r"\n"
            r"\*\*Chapter Summary:\*\*\n"
            + PLANNED_CHAPTER_SUMMARY_RE
        )

    return r"^" + r"\n\n".join(later_chapter_patterns) + r"\n?$"


def build_full_book_chapter_plan_step_regex(
    planned_chapter: PlannedChapterInfo,
    *,
    include_leading_separator: bool,
) -> str:
    leading_separator = r"\n\n" if include_leading_separator else ""
    chapter_pattern = (
        r"### " + _escape_regex_literal(planned_chapter["chapter_name"]) + r"\n"
        r"\*\*Word Count:\*\* " + _escape_regex_literal(str(planned_chapter["chapter_word_count"])) + r"\n"
        r"\*\*Embedding Space:\*\* " + _escape_regex_literal(planned_chapter["chapter_embedding_space"]) + r"\n"
        r"\*\*Narrative Focuses:\*\* " + NARRATIVE_FOCUSES_RE + r"\n"
        r"\*\*Chapter Summary:\*\*\n"
        + PLANNED_CHAPTER_SUMMARY_RE
    )
    return r"^" + leading_separator + chapter_pattern + r"\n?$"


def build_full_book_chapter_plan_continuation_regex() -> str:
    return (
        r"^\n\*\*Narrative Focuses:\*\* "
        + NARRATIVE_FOCUSES_RE
        + r"\n"
        r"\*\*Chapter Summary:\*\*\n"
        + PLANNED_CHAPTER_SUMMARY_RE
        + r"\n?$"
    )
