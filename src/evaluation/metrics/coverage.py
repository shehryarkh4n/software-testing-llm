#!/usr/bin/env python3
"""
coverage.py  – stand-alone metric script

Usage:
    python coverage.py  --tests_json tests.json  --focal_methods focal.json

The two JSON files look like:

    # tests.json  →  list[str]   (each str is one Java test file)
    [
      "import org.junit.Test; ... class GeneratedTest { ... }",
      "import org.junit.Test; ... class GeneratedTest { ... }",
      ...
    ]

    # focal.json  → str          (Java source for the SUT)
    "package my.pkg; public class FocalClass { ... }"

Environment variables (override the defaults if your jars live elsewhere):

    JUNIT_JAR        
    JACOCO_AGENT_JAR 
    JACOCO_CLI_JAR   
"""

import argparse, json, os, re, shutil, subprocess, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
from datasets import load_dataset

# --------------------------------------------------------------------------- #
# ─── CONFIG ───────────────────────────────────────────────────────────────── #
# --------------------------------------------------------------------------- #
HOME = os.environ.get("HOME", "")
JUNIT_JAR        = f"{HOME}/software-testing-llm/src/test-artifacts/junit-4.13.2.jar:{HOME}/software-testing-llm/src/test-artifacts/hamcrest-core-1.3.jar"
JACOCO_AGENT_JAR = f"{HOME}/software-testing-llm/src/test-artifacts/jacocoagent.jar"
JACOCO_CLI_JAR   = f"{HOME}/software-testing-llm/src/test-artifacts/jacococli.jar"

CLASS_RE    = re.compile(r"\bclass\s+GeneratedTest\b")
PACKAGE_RE  = re.compile(r"^\s*package\s+([\w\.]+)\s*;", re.MULTILINE)
CLASS_DECL_RE = re.compile(r"\bclass\s+(\w+)\b")

# --------------------------------------------------------------------------- #
# ─── UTILS ────────────────────────────────────────────────────────────────── #
# --------------------------------------------------------------------------- #

def uniquify_test(idx: int, src: str) -> tuple[str, Path]:
    """
    Rename 'class GeneratedTest' → 'class GenTest<idx>' and
    return (new_source, rel_path) where rel_path is the file
    location respecting any original package declaration.
    Also ensure proper imports are present.
    """
    cls_name = f"GenTest{idx}"
    src = CLASS_RE.sub(f"class {cls_name}", src, count=1)

    # Check and add missing imports
    if '@Test' in src and 'import org.junit.Test;' not in src:
        # Find where to insert the import
        if 'import ' in src:
            # Add after existing imports
            imports = []
            lines = src.split('\n')
            import_end_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import '):
                    import_end_idx = i
                elif line.strip() and not line.strip().startswith('import ') and import_end_idx > 0:
                    break
            
            # Insert the missing import
            lines.insert(import_end_idx + 1, 'import org.junit.Test;')
            src = '\n'.join(lines)
        else:
            # Add import at the beginning
            src = 'import org.junit.Test;\n' + src

    m = PACKAGE_RE.search(src)
    if m:
        pkg_path = Path(*m.group(1).split("."))
        rel_file = pkg_path / f"{cls_name}.java"
    else:
        rel_file = Path(f"{cls_name}.java")

    return src, rel_file

def fix_focal_class_syntax(src: str) -> str:
    """
    Fix the focal class syntax issues:
    1. Add 'public class' if missing
    2. Remove method signatures and replace with implementations
    3. Handle generic methods properly
    """
    # Step 1: Ensure proper class declaration
    src = ensure_class_keyword(src)
    
    # Step 2: Parse the content more carefully
    # The source appears to be all on one line, so let's work with that
    
    # First, let's break down the class structure
    # Find the class opening brace
    class_match = re.search(r'(public class \w+)\s*\{(.+)\}', src)
    if not class_match:
        raise RuntimeError("Cannot parse class structure")
    
    class_decl = class_match.group(1)
    class_body = class_match.group(2).strip()
    
    # Split the body by looking for method boundaries
    # We need to carefully separate implementations from signatures
    
    # Find all method implementations (those with { ... })
    implementations = []
    impl_pattern = r'((?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:<[^>]*>\s*)?[\w<>\[\]]+\s+\w+\s*\([^)]*\))\s*\{([^}]*)\}'
    
    for match in re.finditer(impl_pattern, class_body):
        full_impl = match.group(0)
        implementations.append(full_impl)
    
    # Remove all implementations from the body to get signatures
    body_without_impls = class_body
    for impl in implementations:
        body_without_impls = body_without_impls.replace(impl, '')
    
    # Now parse signatures (things ending with ;)
    signatures = []
    remaining = body_without_impls.strip()
    
    # Split by semicolons and clean up
    parts = remaining.split(';')
    for part in parts:
        part = part.strip()
        if part and ('(' in part or 'GeneratedAttributeSupport()' in part):
            signatures.append(part + ';')
    
    # Now create implementations for all signatures
    new_implementations = []
    
    for sig in signatures:
        sig = sig.replace(';', '').strip()
        
        if 'GeneratedAttributeSupport()' in sig:
            # Constructor
            new_implementations.append('public GeneratedAttributeSupport() {}')
        elif 'valueOf' in sig:
            # Method signature - create implementation
            if 'List<T>' in sig and 'T[]' in sig:
                new_implementations.append('static <T> List<T> valueOf(T[] value) { return java.util.Arrays.asList(value); }')
            elif 'List<T>' in sig and 'List<T>' in sig and 'T[]' not in sig:
                new_implementations.append('static <T> List<T> valueOf(List<T> value) { return value; }')
            elif sig.endswith('T valueOf(T value)'):
                new_implementations.append('static <T> T valueOf(T value) { return value; }')
            elif 'List<Byte>' in sig and 'byte[]' in sig:
                new_implementations.append('static List<Byte> valueOf(byte[] value) { List<Byte> list = new java.util.ArrayList<>(); for(byte b : value) list.add(b); return list; }')
            elif 'List<Short>' in sig and 'short[]' in sig:
                new_implementations.append('static List<Short> valueOf(short[] value) { List<Short> list = new java.util.ArrayList<>(); for(short s : value) list.add(s); return list; }')
            elif 'List<Integer>' in sig and 'int[]' in sig:
                new_implementations.append('static List<Integer> valueOf(int[] value) { List<Integer> list = new java.util.ArrayList<>(); for(int i : value) list.add(i); return list; }')
            elif 'List<Long>' in sig and 'long[]' in sig:
                new_implementations.append('static List<Long> valueOf(long[] value) { List<Long> list = new java.util.ArrayList<>(); for(long l : value) list.add(l); return list; }')
            elif 'List<Float>' in sig and 'float[]' in sig:
                new_implementations.append('static List<Float> valueOf(float[] value) { List<Float> list = new java.util.ArrayList<>(); for(float f : value) list.add(f); return list; }')
            elif 'List<Double>' in sig and 'double[]' in sig:
                new_implementations.append('static List<Double> valueOf(double[] value) { List<Double> list = new java.util.ArrayList<>(); for(double d : value) list.add(d); return list; }')
            elif 'List<Boolean>' in sig and 'boolean[]' in sig:
                new_implementations.append('static List<Boolean> valueOf(boolean[] value) { List<Boolean> list = new java.util.ArrayList<>(); for(boolean b : value) list.add(b); return list; }')
            elif 'List<Character>' in sig and 'char[]' in sig:
                new_implementations.append('static List<Character> valueOf(char[] value) { List<Character> list = new java.util.ArrayList<>(); for(char c : value) list.add(c); return list; }')
            elif 'Byte valueOf(byte' in sig:
                new_implementations.append('static Byte valueOf(byte value) { return value; }')
            elif 'Short valueOf(short' in sig:
                new_implementations.append('static Short valueOf(short value) { return value; }')
            elif 'Integer valueOf(int' in sig:
                new_implementations.append('static Integer valueOf(int value) { return value; }')
            elif 'Long valueOf(long' in sig:
                new_implementations.append('static Long valueOf(long value) { return value; }')
            elif 'Float valueOf(float' in sig:
                new_implementations.append('static Float valueOf(float value) { return value; }')
            elif 'Double valueOf(double' in sig:
                new_implementations.append('static Double valueOf(double value) { return value; }')
            elif 'Boolean valueOf(boolean' in sig:
                new_implementations.append('static Boolean valueOf(boolean value) { return value; }')
            elif 'Character valueOf(char' in sig:
                new_implementations.append('static Character valueOf(char value) { return value; }')
    
    # Remove duplicates from implementations (keep existing ones)
    existing_sigs = set()
    for impl in implementations:
        # Extract signature from implementation
        match = re.match(r'(.*?)\s*\{', impl)
        if match:
            sig = match.group(1).strip()
            sig = re.sub(r'\s+', ' ', sig)
            existing_sigs.add(sig)
    
    final_implementations = list(implementations)  # Keep existing implementations
    
    # Add new implementations that don't conflict
    for new_impl in new_implementations:
        match = re.match(r'(.*?)\s*\{', new_impl)
        if match:
            sig = match.group(1).strip()
            sig = re.sub(r'\s+', ' ', sig)
            # Check if this signature already exists
            conflict = False
            for existing_sig in existing_sigs:
                if signatures_match(sig, existing_sig):
                    conflict = True
                    break
            if not conflict:
                final_implementations.append(new_impl)
                existing_sigs.add(sig)
    
    # Reconstruct the class
    imports = 'import java.util.List;\nimport java.util.ArrayList;\nimport java.util.Arrays;\n\n'
    class_body_final = '\n    ' + '\n    '.join(final_implementations) + '\n'
    result = imports + class_decl + ' {' + class_body_final + '}'
    
    return result

def signatures_match(sig1: str, sig2: str) -> bool:
    """Check if two method signatures match (same method)"""
    # Extract method name and parameters
    def extract_method_core(sig):
        # Remove modifiers and return type, keep method name and params
        match = re.search(r'(\w+)\s*\(([^)]*)\)', sig)
        if match:
            name = match.group(1)
            params = re.sub(r'\s+', '', match.group(2))  # normalize params
            return f"{name}({params})"
        return sig
    
    return extract_method_core(sig1) == extract_method_core(sig2)



def ensure_class_keyword(src: str) -> str:
    """
    Turn  'DateUtilities { … }'
    into  'public class DateUtilities { … }'
    (If 'class' is already present, return unchanged.)
    
    Also handles cases like 'ClassName implements Interface { ... }'
    """
    head = src.split("{", 1)[0]
    if "class" in head:
        return src                                  # already fine

    # Look for class name pattern, possibly with implements/extends
    # Pattern: ClassName [implements Interface] [extends SuperClass] {
    m = re.match(r"\s*([A-Z]\w*)(\s+(?:implements|extends)\s+[^{]*)?[\s]*\{", src)
    if not m:
        raise RuntimeError(f"Cannot find class name in focal source. Source starts with: {src[:100]}...")
    
    class_name = m.group(1)
    class_suffix = m.group(2) if m.group(2) else ""  # implements/extends clause
    
    # Replace the beginning with proper class declaration
    return re.sub(r"^\s*([A-Z]\w*)(\s+(?:implements|extends)\s+[^{]*)?[\s]*\{",
                  f"public class {class_name}{class_suffix} {{",
                  src, count=1)

def write_sources(work: Path, focal_src: str, tests: list[str]) -> tuple[Path, Path]:
    """
    Layout:
        work/
          src/         (production)
          tests/       (generated tests)
          classes/     (.class files of SUT)
          test-classes/(.class files of tests)
    """
    src_dir   = work / "src"
    test_dir  = work / "tests"
    cls_dir   = work / "classes"
    tcls_dir  = work / "test-classes"
    for d in (src_dir, test_dir, cls_dir, tcls_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── focal file ──────────────────────────────────────────────────────────
    # Fix the focal source syntax first
    focal_src = fix_focal_class_syntax(focal_src)
    
    # 1.  class name
    m_cls = CLASS_DECL_RE.search(focal_src)

    if not m_cls:
        raise RuntimeError("No 'class X' declaration found in focal source")
    cls_name = m_cls.group(1)

    # 2.  optional package → directory
    m_pkg = PACKAGE_RE.search(focal_src)
    pkg_path = Path(*m_pkg.group(1).split(".")) if m_pkg else Path()

    focal_file = src_dir / pkg_path / f"{cls_name}.java"
    focal_file.parent.mkdir(parents=True, exist_ok=True)
    focal_file.write_text(focal_src)

    # tests
    for i, raw in enumerate(tests):
        fixed_src, rel_path = uniquify_test(i, raw)
        file_path = test_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(fixed_src)

    return src_dir, test_dir

def javac_compile(source_dir: Path, out_dir: Path, extra_cp: str = ""):
    java_files = [str(p) for p in source_dir.rglob("*.java")]
    if not java_files:
        raise RuntimeError(f"No .java files under {source_dir}")

    # build the class-path list, then stringify every part
    cp_parts = [out_dir]            # a Path
    if extra_cp:
        cp_parts.append(extra_cp)   # a str

    cp_flag = os.pathsep.join(str(p) for p in cp_parts if p)   # ← fix here
    cmd = ["javac", "-d", str(out_dir), "-cp", cp_flag] + java_files
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print(f"Command: {' '.join(cmd)}")
        raise

def discover_test_classes(tcls_dir: Path) -> list[str]:
    """
    Return fully-qualified class names under test-classes/ .
    """
    classes = []
    for p in tcls_dir.rglob("*.class"):
        rel = p.relative_to(tcls_dir).with_suffix("")          # strip .class
        classes.append(".".join(rel.parts))                    # / → .
    return sorted(classes)

def run_tests_with_jacoco(work: Path, cls_dir: Path, tcls_dir: Path) -> Path:
    exec_file = work / "jacoco.exec"
    full_cp   = ":".join([JUNIT_JAR, str(cls_dir), str(tcls_dir)])

    test_classes = discover_test_classes(tcls_dir)
    if not test_classes:
        raise RuntimeError("No compiled test classes found")

    cmd = [
        "java",
        f"-javaagent:{JACOCO_AGENT_JAR}=destfile={exec_file},append=false",
        "-cp", full_cp,
        "org.junit.runner.JUnitCore",
        *test_classes
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Test execution failed:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    
    return exec_file

def jacoco_report(work: Path, exec_file: Path, cls_dir: Path, src_dir: Path) -> Path:
    xml_report = work / "coverage.xml"
    cmd = [
        "java", "-jar", JACOCO_CLI_JAR, "report", str(exec_file),
        f"--classfiles={cls_dir}",
        f"--sourcefiles={src_dir}",
        f"--xml={xml_report}"
    ]
    subprocess.run(cmd, check=True)
    return xml_report

def extract_focal_method_name(focal_src: str) -> str:
    """
    Extract the focal method name from the focal source code.
    Looks for method implementations (with body) rather than just signatures.
    """
    print(f"DEBUG: Extracting method name from focal source:")
    print(f"DEBUG: Focal source snippet: {focal_src[:200]}...")
    
    # Look for method implementations (methods with { ... })
    # Pattern: [modifiers] returnType methodName(params) { ... }
    method_impl_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:<[^>]*>\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{'
    
    matches = re.findall(method_impl_pattern, focal_src)
    print(f"DEBUG: All method implementations found: {matches}")
    
    # Filter out constructors (method name same as class name)
    class_name_match = re.search(r'class\s+(\w+)', focal_src)
    class_name = class_name_match.group(1) if class_name_match else ""
    print(f"DEBUG: Class name detected: '{class_name}'")
    
    focal_methods = [m for m in matches if m != class_name and not m.startswith('get') and not m.startswith('set')]
    print(f"DEBUG: Filtered focal methods: {focal_methods}")
    
    if focal_methods:
        selected_method = focal_methods[0]
        print(f"DEBUG: Selected focal method: '{selected_method}'")
        return selected_method
    
    # Fallback: return any method found
    if matches:
        selected_method = matches[0]
        print(f"DEBUG: Using fallback method: '{selected_method}'")
        return selected_method
    
    print("DEBUG: No methods found, returning 'unknownMethod'")
    return "unknownMethod"

def focal_coverage(xml_report: Path, focal_src: str, debug: bool = True) -> float:
    """
    Extract coverage for the focal method by dynamically detecting the method name.
    For overloaded methods, return the highest coverage found.
    """
    tree = ET.parse(xml_report)
    
    # Extract the focal method name from the source
    focal_method_name = extract_focal_method_name(focal_src)
    
    if debug:
        print(f"DEBUG: Looking for focal method: '{focal_method_name}'")
        print(f"DEBUG: All methods found in coverage report:")
        for method in tree.iterfind(".//method"):
            method_name = method.get("name", "")
            counter = method.find("./counter[@type='LINE']")
            if counter is not None:
                covered = int(counter.get("covered", 0))
                missed = int(counter.get("missed", 0))
                print(f"  - Method: '{method_name}', Lines covered: {covered}, missed: {missed}")
            else:
                print(f"  - Method: '{method_name}', No line counter found")
    
    # Look for ALL methods with the focal method name and find the best coverage
    best_coverage = 0.0
    total_covered = 0
    total_missed = 0
    methods_found = 0
    
    for method in tree.iterfind(".//method"):
        method_name = method.get("name", "")
        if method_name == focal_method_name:
            counter = method.find("./counter[@type='LINE']")
            if counter is not None:
                covered = int(counter.get("covered", 0))
                missed = int(counter.get("missed", 0))
                
                # Track totals for potential aggregation
                total_covered += covered
                total_missed += missed
                methods_found += 1
                
                # Track best individual method coverage
                if covered + missed > 0:
                    method_coverage = 100.0 * covered / (covered + missed)
                    best_coverage = max(best_coverage, method_coverage)
    
    if methods_found > 0:
        # Use the method with the highest coverage (the one that was actually executed)
        if debug:
            aggregate_coverage = 100.0 * total_covered / (total_covered + total_missed) if (total_covered + total_missed) > 0 else 0.0
            print(f"DEBUG: Found {methods_found} methods named '{focal_method_name}'")
            print(f"DEBUG: Best individual coverage: {best_coverage:.1f}%")
            print(f"DEBUG: Aggregate coverage: {aggregate_coverage:.1f}% ({total_covered}/{total_covered + total_missed} total lines)")
        
        return best_coverage
    
    if debug:
        print(f"DEBUG: Focal method '{focal_method_name}' not found, trying fallback...")
    
    # Fallback: try to find any implemented method (not constructor)
    class_name_match = re.search(r'class\s+(\w+)', focal_src)
    class_name = class_name_match.group(1) if class_name_match else ""
    
    for method in tree.iterfind(".//method"):
        method_name = method.get("name", "")
        # Skip constructors, getters, setters, and common utility methods
        if (method_name != class_name and 
            method_name != "<init>" and 
            method_name != "<clinit>" and
            not method_name.startswith("get") and 
            not method_name.startswith("set") and
            method_name not in ["toString", "equals", "hashCode"]):
            
            counter = method.find("./counter[@type='LINE']")
            if counter is not None:
                covered = int(counter.get("covered", 0))
                missed = int(counter.get("missed", 0))
                if covered + missed > 0:  # Only count methods that have actual code
                    coverage = 100.0 * covered / (covered + missed)
                    if debug:
                        print(f"DEBUG: Using fallback method '{method_name}': {covered}/{covered+missed} lines covered ({coverage:.1f}%)")
                    return coverage
    
    if debug:
        print("DEBUG: No suitable method found for coverage calculation")
    return 0.0  # method not found

def load_focal_methods(path: Path, valid_indexes: list[int] = None) -> list[str]:
    """Load focal methods, optionally filtered by valid indexes"""
    ds = load_dataset("json", data_files={"x": str(path)})["x"]
    focal_methods = ds["src_fm_fc_ms_ff"]
    
    if valid_indexes is not None:
        focal_methods = [focal_methods[i] for i in valid_indexes if i < len(focal_methods)]
    
    return focal_methods

def load_tests(path: Path) -> tuple[list[str], list[int]]:
    """Load tests and return (tests, original_indexes)"""
    ds = load_dataset("json", data_files={"x": str(path)})["x"]

    tests = ds["prediction"]
    indexes = ds["original_idx"]
    return tests, indexes

# --------------------------------------------------------------------------- #
# ─── MAIN ─────────────────────────────────────────────────────────────────── #
# --------------------------------------------------------------------------- #

def compute_coverage(tests_json: Path, focal_json: Path, save_reports_dir: Path = None):
    tests, idxs   = load_tests(tests_json)             # list[str], list[int]
    focals        = load_focal_methods(focal_json, idxs)

    assert len(tests) == len(focals)
    tests, focals, idxs = tests[:1], focals[:1], idxs[:1]
    # Create save directory if specified
    if save_reports_dir:
        save_reports_dir.mkdir(parents=True, exist_ok=True)

    results = []                                       # (original_idx, coverage)
    for orig_idx, test_src, focal_src in zip(idxs, tests, focals):
        try:
            with tempfile.TemporaryDirectory() as td:
                work   = Path(td)
                src_d, tst_d = write_sources(work, focal_src, [test_src])

                cls_d  = work / "classes"
                tcls_d = work / "test-classes"

                javac_compile(src_d, cls_d)
                javac_compile(tst_d, tcls_d, extra_cp=JUNIT_JAR + ":" + str(cls_d))

                exec_f = run_tests_with_jacoco(work, cls_d, tcls_d)
                xml_r  = jacoco_report(work, exec_f, cls_d, src_d)

                # Save XML report if directory is specified
                if save_reports_dir:
                    saved_report_path = save_reports_dir / f"coverage_test_{orig_idx}.xml"
                    shutil.copy2(xml_r, saved_report_path)
                    print(f"Saved coverage report for test {orig_idx} to {saved_report_path}")

                cov = focal_coverage(xml_r, focal_src)
                results.append((orig_idx, cov))
                
        except Exception as e:
            print(f"Error processing test {orig_idx}: {e}")
            results.append((orig_idx, 0.0))  # Zero coverage on error

    return results, ['']          # list of per-test coverages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests_json", "-t", type=str, required=True)
    parser.add_argument("--focal_methods", "-f", type=str, required=True)
    parser.add_argument("--save_reports", "-s", type=str, default=None, 
                       help="Directory to save XML coverage reports")
    args = parser.parse_args()

    save_dir = Path(args.save_reports) if args.save_reports else None
    score, l = compute_coverage(Path(args.tests_json), Path(args.focal_methods), save_dir)
    print(score)