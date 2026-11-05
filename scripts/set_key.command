#!/bin/zsh
# Saves the OpenRouter key in the macOS Keychain. Nothing is stored in a file or shown on screen.
SERVICE=jev-benchmark-openrouter
clear
echo "Paste your OpenRouter key and press Return."
echo "Nothing appears as you paste. That's normal."
echo "It asks twice, so paste it both times."
echo
security add-generic-password -U -a "$USER" -s "$SERVICE" -w
if [ $? -ne 0 ]; then
  echo; echo "That didn't save. The two pastes may not have matched. Close this window and try again."
else
  echo; echo "Saved. Checking it with OpenRouter (free)..."; echo
  zsh "${0:A:h}/check_key.sh"
  echo; echo "Done. You can close this window."
fi
