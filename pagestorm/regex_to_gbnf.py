"""Compile the (regular) subset of Python regex used by PageStorm stages to GBNF.

The PageStorm stage patterns are regular: literals, character classes,
alternation, grouping, and bounded/unbounded repetition (`{m,n}`, `*`, `+`,
`?`). No backreferences, lookaround, or named groups. That maps cleanly onto
llama.cpp GBNF, whose grammar parser supports `{m,n}` repetition.

`regex_to_gbnf(pattern)` returns a complete grammar with a `root` rule.

Supported regex constructs:
  ^ $ \\Z anchors (dropped — GBNF root matches the whole string)
  literals, escaped literals (\\* \\. \\( \\  ...)
  .                    -> any char except newline
  \\d \\D \\w \\W \\s \\S   -> equivalent character classes
  [..] [^..] (with \\d \\w \\s etc. expanded inside)
  ( ... )  (?: ... )   -> group
  | alternation
  * + ? {n} {n,} {n,m}  (a trailing lazy '?' is accepted and ignored)
"""

from __future__ import annotations


# llama.cpp expands `{m,n}` by *unrolling* into n copies. The real blow-up came
# from *inlined nested groups* (a `{1,90}` group containing a `{1,300}` element
# inlines the 300-unroll inside all 90 copies -> multiplicative). We fix that by
# emitting every group as a named rule (see `_Parser._new_rule`), so nested
# repetition unrolls into cheap rule *references* and each inner unroll is
# defined once. Exact bounds are therefore preserved. This threshold only guards
# against pathologically huge single bounds; the PageStorm patterns top out at
# 300, well under it.
MAX_UNROLL = 400


def _bound_repetition(body: str) -> str:
    body = body.strip()
    if "," not in body:
        # exact {n}: keep (these are small in the PageStorm patterns)
        return body
    lo, _, hi = body.partition(",")
    lo = lo.strip()
    hi = hi.strip()
    if hi == "":
        return f"{lo or 0},"  # already unbounded
    if int(hi) > MAX_UNROLL:
        return f"{lo or 0},"  # safety net: drop a pathological upper bound
    return f"{lo or 0},{hi}"


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.s = pattern
        self.i = 0
        self.n = len(pattern)
        self.rules: list[str] = []
        self._counter = 0

    def _new_rule(self, body: str) -> str:
        self._counter += 1
        name = f"r{self._counter}"
        self.rules.append(f"{name} ::= {body}")
        return name

    def peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def next(self) -> str:
        ch = self.s[self.i]
        self.i += 1
        return ch

    def eof(self) -> bool:
        return self.i >= self.n

    # alternation -> sequence ('|' sequence)*
    def parse_alternation(self) -> str:
        branches = [self.parse_sequence()]
        while self.peek() == "|":
            self.next()
            branches.append(self.parse_sequence())
        if len(branches) == 1:
            return branches[0]
        empty = '""'
        return "(" + " | ".join(b or empty for b in branches) + ")"

    def parse_sequence(self) -> str:
        parts: list[str] = []
        pending_literal: list[str] = []

        def flush_literal() -> None:
            if pending_literal:
                parts.append(_gbnf_string("".join(pending_literal)))
                pending_literal.clear()

        while not self.eof() and self.peek() not in "|)":
            ch = self.peek()
            if ch in "^$":
                # anchors: drop (root matches whole string)
                self.next()
                continue
            atom, is_literal_char = self.parse_atom()
            if atom is None:
                continue
            quant = self.parse_quantifier()
            if quant is None and is_literal_char:
                # accumulate bare literal chars into one quoted run
                pending_literal.append(atom)
                continue
            flush_literal()
            element = _gbnf_string(atom) if is_literal_char else atom
            parts.append(element + (quant or ""))
        flush_literal()
        if not parts:
            return ""
        return " ".join(parts)

    def parse_quantifier(self) -> str | None:
        ch = self.peek()
        if ch and ch in "*+?":
            self.next()
            if self.peek() == "?":  # lazy -> ignore
                self.next()
            return ch
        if ch == "{":
            j = self.s.find("}", self.i)
            if j == -1:
                return None
            body = self.s[self.i + 1 : j]
            self.i = j + 1
            if self.peek() == "?":
                self.next()
            return "{" + _bound_repetition(body) + "}"
        return None

    # returns (gbnf_or_literalchar, is_literal_char)
    def parse_atom(self) -> tuple[str | None, bool]:
        ch = self.next()
        if ch == "(":
            # consume (?: if present
            if self.s[self.i : self.i + 2] == "?:":
                self.i += 2
            inner = self.parse_alternation()
            if self.peek() == ")":
                self.next()
            # Emit the group as a named rule so repetition unrolls into cheap
            # references instead of inlining (and multiplying) its body.
            return self._new_rule(inner or '""'), False
        if ch == "[":
            return self.parse_class(), False
        if ch == ".":
            return "[^\\n]", False
        if ch == "\\":
            esc = self.next()
            if esc in _ANCHOR_ESCAPES:
                # zero-width anchors (\A \Z \z start/end, \b \B word boundary):
                # drop — the GBNF root matches the whole string.
                return None, False
            cls = _ESCAPE_CLASSES.get(esc)
            if cls is not None:
                return cls, False
            return _unescape_char(esc), True
        # plain literal char
        return ch, True

    def parse_class(self) -> str:
        negated = False
        if self.peek() == "^":
            negated = True
            self.next()
        body: list[str] = []
        neg_shorthands: set[str] = set()  # \S \D \W seen inside (they invert under [^...])
        literals: set[str] = set()        # raw literal chars seen (for set algebra)
        while not self.eof() and self.peek() != "]":
            ch = self.next()
            if ch == "\\":
                esc = self.next()
                if esc in ("S", "D", "W"):
                    neg_shorthands.add(esc)
                    continue
                if esc in ("n", "t", "r"):
                    literals.add(_unescape_char(esc))
                expanded = _CLASS_ESCAPE_EXPANSIONS.get(esc)
                if expanded is not None:
                    body.append(expanded)
                else:
                    body.append(_class_escape_literal(esc))
                    literals.add(esc)
            else:
                body.append(_class_escape_literal(ch))
                literals.add(ch)
        if self.peek() == "]":
            self.next()

        if neg_shorthands:
            # A shorthand like \S (= non-whitespace) inside a class can't be
            # written in GBNF directly, and inside a NEGATED class it inverts.
            # Resolve the common, finite cases to an exact positive class.
            # narrower-but-safe is fine (output stays a subset of the regex).
            return self._resolve_shorthand_class(negated, neg_shorthands, literals)

        return "[" + ("^" if negated else "") + "".join(body) + "]"

    @staticmethod
    def _resolve_shorthand_class(negated: bool, shorthands: set[str], literals: set[str]) -> str:
        # We only need \S (the PageStorm patterns use [^\S\n] = horizontal
        # whitespace, and [^\S] = all whitespace). Others would be ambiguous to
        # collapse safely, so we refuse and let the caller fall back.
        if shorthands != {"S"} or not negated:
            raise ValueError(f"unsupported shorthand-in-class: negated={negated} {shorthands}")
        # [^\S ...] == whitespace minus any whitespace chars explicitly listed.
        ws = [" ", "\\t", "\\r", "\\n", "\\f"]
        listed_ws = {"\n", "\t", "\r"} & literals
        keep = [w for w in ws if not (w == "\\n" and "\n" in listed_ws)
                and not (w == "\\t" and "\t" in listed_ws)
                and not (w == "\\r" and "\r" in listed_ws)]
        return "[" + "".join(keep) + "]"


# Zero-width anchors -> dropped (root matches whole string)
_ANCHOR_ESCAPES = {"A", "Z", "z", "b", "B"}

# \X outside a class -> GBNF char class
_ESCAPE_CLASSES = {
    "d": "[0-9]",
    "D": "[^0-9]",
    "w": "[0-9A-Za-z_]",
    "W": "[^0-9A-Za-z_]",
    "s": "[ \\t\\r\\n]",
    "S": "[^ \\t\\r\\n]",
}

# \X inside a class -> raw class fragment (no brackets)
_CLASS_ESCAPE_EXPANSIONS = {
    "d": "0-9",
    "w": "0-9A-Za-z_",
    "s": " \\t\\r\\n",
    "n": "\\n",
    "t": "\\t",
    "r": "\\r",
}


def _unescape_char(esc: str) -> str:
    return {"n": "\n", "t": "\t", "r": "\r"}.get(esc, esc)


def _class_escape_literal(ch: str) -> str:
    # Characters that are special inside a GBNF [...] class.
    # NOTE: '-' is intentionally NOT escaped — these patterns use it only for
    # ranges ([0-9], [1-9], [A-Za-z]); escaping it would break the range into a
    # literal char set.
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == "\r":
        return "\\r"
    if ch in "]\\":
        return "\\" + ch
    return ch


def _gbnf_string(text: str) -> str:
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def regex_to_gbnf(pattern: str, root_name: str = "root") -> str:
    parser = _Parser(pattern)
    body = parser.parse_alternation()
    if not parser.eof():
        raise ValueError(f"regex_to_gbnf: unparsed tail at {parser.i}: {pattern[parser.i:parser.i+30]!r}")
    if not body:
        body = '""'
    return "\n".join([f"{root_name} ::= {body}"] + parser.rules)


if __name__ == "__main__":
    import sys
    print(regex_to_gbnf(sys.stdin.read().rstrip("\n")))
