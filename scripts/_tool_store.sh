# Sourced by scripts/run_tests.sh, never executed: which tool store the
# runner's own provisioning (run-in-hermes-env -> setup-hermes.sh -> pm install)
# writes. Callers set REPO_ROOT and source scripts/_activation.sh first
# (hermes_bootstrap_python).
#
# The account's tool store (~/.hermes/tools of the OS account) keeps ONE
# facts.json entry per package (scripts/tool_store_pins.py), so a run that
# provisions other pins into it re-points every install that uses it.
#
#   _account_tool_store  faked HOME: borrow the account store when it already
#                        records this checkout's pins (prints account, minor)
#   _private_tool_store  real HOME, a checkout that is not the installed one and
#                        whose pins the account store lacks: a private store per
#                        pin set (prints its path; t_4f98fc8f)

# The OS account's home directory, whatever HOME says.
_account_home() {
  local account
  account="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [ -n "$account" ] || account="$(dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2>/dev/null | sed -n 's/^NFSHomeDirectory: *//p')"
  printf '%s\n' "$account"
}

# Prints the account home, then the pinned Python minor.
_account_tool_store() {
  local account py_minor
  [ -z "${HERMES_RUNTIME_DIR:-}" ] || return 1
  case "${HERMES_HOME:-}" in "" | "$HOME/.hermes" | "$HOME/.hermes/"*) ;; *) return 1 ;; esac
  [ ! -f "$HOME/.hermes/tools/facts.json" ] || return 1
  account="$(_account_home)"
  [ -n "$account" ] && [ -d "$account" ] && [ ! "$account" -ef "$HOME" ] || return 1
  [ -f "$account/.hermes/tools/facts.json" ] && [ -w "$account/.hermes/tools" ] || return 1
  py_minor="$(_store_has_pins "$account/.hermes/tools")" || return 1
  printf '%s\n%s\n' "$account" "$py_minor"
}

# Under the real HOME the account store is the default store, so a worktree, a
# renovate branch or a `git archive` tree on other pins would re-stage it
# (t_4f98fc8f). Only `hermes update` / source completion write
# install-stamp.json (hermes_cli/source_stamp.py), so its absence marks a
# checkout that is not the installed one; the installed checkout, a checkout on
# the account's pins and a fresh account (no store yet) keep today's layout.
# Prints $HOME/.hermes/cache/tool-stores/<first 16 hex of sha256(pm/lock.json)>:
# one store per pin set, reused by later runs on the same pins.
_private_tool_store() {
  local account lock key
  [ -z "${HERMES_RUNTIME_DIR:-}" ] || return 1
  case "${HERMES_HOME:-}" in "" | "$HOME/.hermes" | "$HOME/.hermes/"*) ;; *) return 1 ;; esac
  account="$(_account_home)"
  [ -n "$account" ] && [ "$account" -ef "$HOME" ] || return 1
  [ -f "$HOME/.hermes/tools/facts.json" ] || return 1
  [ ! -e "$REPO_ROOT/install-stamp.json" ] || return 1
  lock="$REPO_ROOT/pm/lock.json"
  [ -f "$lock" ] || return 1
  ! _store_has_pins "$HOME/.hermes/tools" >/dev/null || return 1
  key="$({ sha256sum "$lock" 2>/dev/null || shasum -a 256 "$lock"; } | cut -c1-16)"
  case "$key" in *[!0-9a-f]* | "") return 1 ;; esac
  [ "${#key}" -eq 16 ] || return 1
  printf '%s\n' "$HOME/.hermes/cache/tool-stores/$key"
}

# Read-only check that the store already records this checkout's pins
# (scripts/tool_store_pins.py says why and what counts); prints the Python minor.
_store_has_pins() {
  local python
  python="$(HERMES_RUNTIME_DIR="$1" hermes_bootstrap_python "$REPO_ROOT")" || return 1
  PYTHONHOME= "$python" -I -B "$REPO_ROOT/scripts/tool_store_pins.py" "$REPO_ROOT" "$1" 2>/dev/null
}
