"""
Shared Java source fixers and helpers used by both the compile gate
and the coverage gate.

Design goals
- Idempotent: calling the same fixer twice is safe.
- Deterministic: same input → same output (no randomness).
- Conservative: only add the minimum required code/imports to compile.

Public API
----------
Tests side
  - sanitize_test(raw: str) -> str
  - ensure_outer_class(src: str, default: str = "GenTest") -> str
  - ensure_std_imports(src: str) -> str
  - uniquify_test(idx: int, src: str, base: str = "GenTest") -> tuple[str, str]
  - make_stubs(tmp_dir: Path, src: str) -> None

Focal side
  - ensure_class_keyword(src: str) -> str
  - normalize_focal(src: str) -> str

General
  - build_classpath(*parts: str | Path) -> str
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# ------------------------------ Regex & constants --------------------------- #
TOP_PUBLIC_RE       = re.compile(r"^\s*public\s+(class|interface|enum)\s+", re.MULTILINE)
CLASS_DECL_RE       = re.compile(r"\bclass\s+([A-Z][A-Za-z0-9_]*)\b")
CLASS_OPEN_RE       = re.compile(r"\bclass\b")
PACKAGE_RE          = re.compile(r"^\s*package\s+([\w\.]+)\s*;", re.MULTILINE)
IMPORT_RE           = re.compile(r'^\s*import\s+(static\s+)?([\w\.]+)(\.[\*])?\s*;', re.MULTILINE)
METHOD_IMPL_RE      = re.compile(r"((?:public|protected|private)\s+)?((?:static)\s+)?(?:<[^>]*>\s*)?([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*\{", re.DOTALL)
METHOD_SIG_RE       = re.compile(r"((?:public|protected|private)\s+)?((?:static)\s+)?(?:<[^>]*>\s*)?([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^;{]+)?;")
BARE_ASSERT_RE      = re.compile(r"\bassert[A-Z]\w*\s*\(")
TEST_ANNOT_RE       = re.compile(r"@\s*Test\b")
METHOD_CALL_RE      = re.compile(r"([A-Z][A-Za-z0-9_]*)\s*\.\s*([a-zA-Z_]\w*)\s*\(")
CAP_ID_RE           = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")

ALLOWED_IMPORT_PREFIXES = ("java.", "javax.", "org.junit.")
JAVA_STD_CLASS_NAMES: Set[str] = {
    "String", "System", "Math", "Date",
    "LocalDate", "LocalTime", "LocalDateTime", "ZoneOffset", "Instant",
    "List", "ArrayList", "Map", "HashMap", "Optional",
    "Test", "Assert", "Before", "After", "BeforeClass", "AfterClass",
}

# ------------------------------ Small utils -------------------------------- #

def build_classpath(*parts: Iterable[str | Path]) -> str:
    """Join classpath parts with the correct OS-specific separator."""
    cleaned: List[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, (list, tuple, set)):
            for q in p:  # flatten 1 level
                if q:
                    cleaned.append(str(q))
        else:
            cleaned.append(str(p))
    return os.pathsep.join(cleaned)

# ------------------------------ Test fixers -------------------------------- #

def sanitize_test(raw: str) -> str:
    """Drop a top-level `public` so filename mismatches don't break compile."""
    if not raw:
        return ""
    return TOP_PUBLIC_RE.sub(r"\1 ", raw, count=1)



def ensure_outer_class(src: str, default: str = "GenTest") -> str:
    """Wrap body in a simple class if no top-level class is present."""
    if CLASS_OPEN_RE.search(src):
        return src
    lines = src.splitlines()
    header_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith(("package", "import")) or ln.strip() == "":
            header_end = i + 1
        else:
            break
    header = lines[:header_end]
    body = lines[header_end:] or [""]
    wrapped = header + [f"class {default} {{"] + body + ["}"]
    return "\n".join(wrapped)


def ensure_std_imports(src: str) -> str:
    """Ensure minimal imports often missing in generated tests."""
    need_time_util = ("import java.time" not in src) and ("LocalDate" in src or "LocalTime" in src or "LocalDateTime" in src or "Instant" in src or "ZoneOffset" in src)
    needs_static = bool(BARE_ASSERT_RE.search(src)) and ("import static org.junit.Assert" not in src)
    needs_test_imp = bool(TEST_ANNOT_RE.search(src)) and ("import org.junit.Test" not in src)

    if not (need_time_util or needs_static or needs_test_imp):
        return src

    lines = src.splitlines()
    class_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("class")), len(lines))

    new_imports: List[str] = []
    if need_time_util:
        new_imports += ["import java.time.*;", "import java.util.*;"]
    if needs_static:
        new_imports.append("import static org.junit.Assert.*;")
    if needs_test_imp:
        new_imports.append("import org.junit.Test;")

    # insert just before first class decl
    lines[class_idx:class_idx] = new_imports
    return "\n".join(lines)


def collect_imports(src: str) -> List[Tuple[bool, str, bool]]:
    """Return list of (is_static, fqcn, is_wildcard)."""
    out: List[Tuple[bool, str, bool]] = []
    for m in IMPORT_RE.finditer(src):
        out.append((bool(m.group(1)), m.group(2), bool(m.group(3))))
    return out


def _needs_stub_import(fqcn: str) -> bool:
    return not fqcn.startswith(ALLOWED_IMPORT_PREFIXES)


def _stub_for_fqcn(tmp_dir: Path, fqcn: str) -> None:
    parts = fqcn.split(".")
    cls = parts[-1]
    if not cls or not cls[0].isalpha():
        return
    pkg = ".".join(parts[:-1])
    pkg_dir = tmp_dir.joinpath(*parts[:-1])
    pkg_dir.mkdir(parents=True, exist_ok=True)
    code = [
        f"package {pkg};" if pkg else "",
        f"public class {cls} {{",
        "    public static Object ANY = null;",
        "    public static <T> T any() { return null; }",
        "    public static <T> T when(Object o) { return null; }",
        "    public static <T> T verify(Object o) { return null; }",
        "    public static <T> T mock(Class<T> c) { return null; }",
        "}",
    ]
    (pkg_dir / f"{cls}.java").write_text("\n".join([l for l in code if l != ""]))


def _declared_class_names(src: str) -> Set[str]:
    return set(CLASS_DECL_RE.findall(src))


def _capitalized_identifiers(src: str) -> Set[str]:
    return set(CAP_ID_RE.findall(src))


def missing_classes(src: str) -> Set[str]:
    declared = _declared_class_names(src)
    ids = _capitalized_identifiers(src)
    return {c for c in ids if c not in JAVA_STD_CLASS_NAMES and c not in declared}


def methods_per_class(src: str) -> Dict[str, Set[str]]:
    calls: Dict[str, Set[str]] = {}
    for cls, m in re.findall(r"([A-Z][A-Za-z0-9_]*)\s*\.\s*([a-zA-Z_]\w*)\s*\(", src):
        calls.setdefault(cls, set()).add(m)
    return calls


def _stub_code(cls: str, methods: Set[str] | None) -> str:
    lines = [
        f"class {cls} {{",
        f"    public {cls}(Object... a) {{}}",
        "    public static Object ANY = null;"
    ]
    for m in (methods or {"any"}):
        lines.append(f"    public static <T> T {m}(Object... a) {{ return null; }}")
    lines.append("}")
    return "\n".join(lines)


def make_stubs(tmp_dir: Path, src: str) -> None:
    # external imports
    for is_static, fqcn, _ in collect_imports(src):
        if _needs_stub_import(fqcn):
            _stub_for_fqcn(tmp_dir, fqcn)
    # class names referenced but missing
    miss = missing_classes(src)
    call_map = methods_per_class(src)
    for cls in miss:
        (tmp_dir / f"{cls}.java").write_text(_stub_code(cls, call_map.get(cls, set())))


def uniquify_test(idx: int, src: str, base: str = "GenTest") -> tuple[str, str]:
    """
    Rename a class named GeneratedTest (or first class) → {base}{idx} and
    compute its relative path honoring original package.
    Returns (new_source, relative_path_str)
    """
    cls_name = f"{base}{idx}"

    # Replace "class GeneratedTest" or first class decl with desired name.
    def _replace_class(m: re.Match[str]) -> str:
        return m.group(0).replace(m.group(1), cls_name, 1)

    m = re.search(r"\bclass\s+(GeneratedTest)\b", src)
    if m:
        src = src[:m.start()] + _replace_class(m) + src[m.end():]
    else:
        # Replace the first declared class name if present (safe enough for tests)
        m2 = CLASS_DECL_RE.search(src)
        if m2:
            src = src[:m2.start(1)] + cls_name + src[m2.end(1):]
        else:
            # no class? ensure one
            src = ensure_outer_class(src, default=cls_name)

    # Ensure JUnit imports if @Test is present
    if "@Test" in src and "import org.junit.Test;" not in src and "import org.junit.*;" not in src:
        # Insert after package / imports
        lines = src.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("import "):
                insert_at = i + 1
            elif line.strip() and not line.strip().startswith(("package", "import")):
                break
        lines.insert(insert_at, "import org.junit.Test;")
        src = "\n".join(lines)

    pkg_match = PACKAGE_RE.search(src)
    if pkg_match:
        rel = Path(*pkg_match.group(1).split(".")) / f"{cls_name}.java"
    else:
        rel = Path(f"{cls_name}.java")
    return src, str(rel)

# ------------------------------ Focal fixers -------------------------------- #

def ensure_class_keyword(src: str) -> str:
    """
    Turn  'DateUtilities { … }' into 'public class DateUtilities { … }'.
    If 'class' already present in header, return unchanged.
    """
    head = src.split("{", 1)[0]
    if "class" in head:
        return src
    m = re.match(r"\s*([A-Z]\w*)(\s+(?:implements|extends)\s+[^\{]*)?\s*\{", src)
    if not m:
        # If we do not even see a '{', wrap everything
        tokens = re.findall(r"[A-Z]\w+", src)
        cname = tokens[0] if tokens else "FocalClass"
        return f"public class {cname} {{\n{src}\n}}"
    cname = m.group(1)
    suffix = m.group(2) or ""
    return re.sub(r"^\s*([A-Z]\w*)(\s+(?:implements|extends)\s+[^\{]*)?\s*\{",
                  f"public class {cname}{suffix} {{", src, count=1)


def _default_return_for_type(ret: str) -> str:
    ret = ret.strip()
    # strip generics to decide primitives/knowns
    base = re.sub(r"<.*>", "", ret)
    if base in {"void"}:
        return ""
    if base in {"byte", "short", "int", "long"}:
        return "return 0;"
    if base in {"float", "double"}:
        return "return 0.0;"
    if base == "boolean":
        return "return false;"
    if base == "char":
        return "return '\\0';"
    # Collections: prefer constructing empties
    if base.endswith("[]"):
        return "return null;"  # simplest and safe for compilation
    if base in {"List", "Collection", "Iterable"} or base.startswith("List") or base.startswith("Collection"):
        return "return new java.util.ArrayList<>();"
    if base in {"Map"} or base.startswith("Map"):
        return "return new java.util.HashMap<>();"
    if base.endswith("Optional") or base.startswith("Optional"):
        return "return java.util.Optional.empty();"
    return "return null;"


def _implement_signature(mod1: str, mod2: str, ret: str, name: str, params: str, throws: str | None) -> str:
    mods = (mod1 or "") + (mod2 or "")
    mods = re.sub(r"\s+", " ", mods).strip()
    prefix = (mods + " ") if mods else ""
    # throws kept verbatim if present
    throws_part = f" throws {throws.strip()}" if throws else ""
    body_return = _default_return_for_type(ret)
    if not body_return and ret.strip() == "void":
        body = "{}"
    else:
        body = "{ " + body_return + " }"
    return f"{prefix}{ret} {name}({params}){throws_part} {body}"


def _normalize_class_body(class_src: str, class_name: str) -> str:
    """
    Replace all method *signatures* (ending with ';') with trivial bodies.
    Keep existing implementations as-is. Ensure imports for java.util if we
    created any collection defaults.
    """
    # Extract throws in signatures
    # We'll re-scan using a throw-aware regex
    sig_with_throws = re.compile(
        r"((?:public|protected|private)\s+)?((?:static)\s+)?(?:<[^>]*>\s*)?([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+([^;{]+))?\s*;"
    )

    impl_spans: List[Tuple[int, int]] = []
    for m in METHOD_IMPL_RE.finditer(class_src):
        # naive brace matching to skip replacing inside impls
        start = m.start()
        # Find matching '}' for this method by simple counter
        i = m.end()
        depth = 1
        while i < len(class_src) and depth > 0:
            if class_src[i] == '{':
                depth += 1
            elif class_src[i] == '}':
                depth -= 1
            i += 1
        impl_spans.append((start, i))

    # Build a mask of impl regions
    keep = [True] * len(class_src)
    for a, b in impl_spans:
        for i in range(a, b):
            keep[i] = False

    # Replace signatures outside impls
    out = []
    i = 0
    while i < len(class_src):
        if keep[i]:
            m = sig_with_throws.match(class_src, i)
            if m:
                mod1, mod2, ret, name, params, throws = m.groups()
                impl = _implement_signature(mod1, mod2, ret, name, params, throws)
                out.append(impl)
                i = m.end()
                continue
        out.append(class_src[i])
        i += 1

    return "".join(out)


def normalize_focal(src: str) -> str:
    """
    Produce a compilable focal class from a variety of inputs:
    - class with bodies (returned mostly unchanged)
    - class with method signatures → add trivial bodies
    - bare method → wrap in a public class
    Also inject minimal imports when our default bodies require them.
    """
    if not src:
        return "public class FocalClass {}"

    # If it looks like a bare method (no 'class' before first '{')
    if 'class' not in src.split('{', 1)[0]:
        # Try to grab a method signature or impl
        m = re.search(r"([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*\{?", src)
        cname = "FocalClass"
        if m:
            # Guess class name from a capitalized identifier near start
            caps = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", src)
            if caps:
                cname = caps[0]
        wrapped = f"public class {cname} {{\n{src}\n}}"
        src = wrapped

    src = ensure_class_keyword(src)

    # Extract class name & body
    m_cls = re.search(r"class\s+(\w+)\s*([^\{]*)\{", src)
    if not m_cls:
        return src  # give up gracefully
    cname = m_cls.group(1)

    # Work only on the portion inside outermost class braces
    start = src.find('{', m_cls.end() - 1)
    if start == -1:
        return src
    # naive match of closing brace
    depth = 1
    i = start + 1
    while i < len(src) and depth > 0:
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
        i += 1
    end = i

    header = src[: start + 1]
    body = src[start + 1 : end - 1]
    tail = src[end - 1 :]

    new_body = _normalize_class_body(body, cname)

    out = header + "\n    " + new_body.strip() + "\n" + tail

    # If our defaults use collections/optional, ensure imports exist
    if ("new java.util.ArrayList" in out or "new java.util.HashMap" in out or "java.util.Optional" in out) and "import java.util" not in out:
        # Insert after package / imports
        lines = out.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("import "):
                insert_at = i + 1
            elif line.strip() and not line.strip().startswith(("package", "import")):
                break
        lines.insert(insert_at, "import java.util.*;")
        out = "\n".join(lines)

    return out

# ------------------------------ End of module ------------------------------- #
