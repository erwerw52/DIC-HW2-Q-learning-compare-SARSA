#!/bin/bash

echo "🔄 [1/4] Preparing to wrap up development session..."

# Write handover document in markdown for next development
echo "📝 [2/4] Generating handover document (handover.md)..."
cat > handover.md << EOF
# Handover Document

**Last Updated:** $(date)

## What was completed in this session:
- Update with your recent achievements here.

## Next steps & pending work:
- Update with tasks for the next developer/session.

## Blockers or Open Questions:
- Note down any blockers here.
EOF

echo "✅ handover.md generated. Please open and edit it before pushing if you want to provide more details."
read -p "Press [Enter] to continue..."

# Update tasks.md
echo "You can manually update your active tasks (openspec/changes/*/tasks.md) or mark them as completed."

# Archive the change if everything is complete
read -p "Is the current OpenSpec change complete and ready to be archived? (y/N): " ARCHIVE_ANS
if [[ "$ARCHIVE_ANS" == "y" || "$ARCHIVE_ANS" == "Y" ]]; then
    echo "📦 Archiving the change..."
    # Assuming OpenSpec has a command for this. Otherwise, a manual move can be done.
    # openspec archive
    # Just generic message for now
    echo "You can move your change folder from openspec/changes/ into openspec/changes/archive/"
fi

# Push code to github
echo "🚀 [3/4] Committing code to GitHub..."
git add .
read -p "Enter commit message (or press enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="chore: Wrap up development session and update handover document"
fi

git commit -m "$COMMIT_MSG"

echo "☁️ [4/4] Pushing code..."
git push origin main

echo "🎉 Code pushed successfully. Development session ended!"