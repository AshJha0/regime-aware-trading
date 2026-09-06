#!/usr/bin/env bash
# Run every *Test class with JUnitCore (bash test.sh from java/, after bash build.sh).
set -euo pipefail
cd "$(dirname "$0")"

JUNIT=/usr/share/java/junit4.jar
HAMCREST=/usr/share/java/hamcrest.jar

TESTS=$(find out/test -name '*Test.class' \
    | sed -e 's|^out/test/||' -e 's|\.class$||' -e 's|/|.|g' | sort)

java -cp "out/main:out/test:$JUNIT:$HAMCREST" org.junit.runner.JUnitCore $TESTS
