#!/bin/sh
# Installs the repo's git hooks into .git/hooks (not itself version-controlled by git).
# Run once after cloning: sh 00-meta/install-hooks.sh
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK" <<'EOF'
#!/bin/sh
# Blocks commits that break intra-repo markdown links or SUMMARY.md nav.
python3 "$(git rev-parse --show-toplevel)/00-meta/check-links.py"
EOF

chmod +x "$HOOK"
echo "Installed pre-commit hook -> $HOOK"
