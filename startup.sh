#!/bin/bash

echo "🚀 [1/4] Pulling latest code from GitHub..."
git pull origin main

echo "📦 [2/4] Initializing OpenSpec..."
openspec init

echo "📄 [3/4] Reading handover document..."
if [ -f "handover.md" ]; then
    echo "=================================================="
    echo "🔽 Handover Document (handover.md)"
    echo "=================================================="
    cat handover.md
    echo "=================================================="
else
    echo "⚠️ No handover.md found. A new one can be created when ending the session."
fi

echo "🎯 [4/4] Suggested next actions:"
if ls openspec/changes/*/tasks.md 1> /dev/null 2>&1; then
    echo "Current active tasks:"
    cat openspec/changes/*/tasks.md
else
    echo "No active tasks found in openspec/changes/. Please check your project board or create a new OpenSpec proposal."
fi

echo "✅ Development environment is ready. Happy coding!"
