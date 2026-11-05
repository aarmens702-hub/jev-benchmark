#!/bin/zsh
# Checks the saved OpenRouter key. Prints only its limit and usage, never the key.
SERVICE=jev-benchmark-openrouter
if ! security find-generic-password -a "$USER" -s "$SERVICE" >/dev/null 2>&1; then
  echo "NO KEY SAVED"; exit 1
fi
printf 'Authorization: Bearer %s' "$(security find-generic-password -a "$USER" -s "$SERVICE" -w)" \
  | curl -s --max-time 20 -H @- https://openrouter.ai/api/v1/key \
  | python3 -I -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print("KEY CHECK FAILED: unreadable response"); sys.exit(1)
if "data" not in d:
    print("KEY CHECK FAILED: OpenRouter rejected it, code", d.get("error",{}).get("code")); sys.exit(1)
d=d["data"]
print("KEY WORKS | limit:",d.get("limit"),"| used:",d.get("usage"))
if d.get("limit") is None: print("WARNING: this key has no spend limit. Set one to 5 on the OpenRouter keys page.")
'
