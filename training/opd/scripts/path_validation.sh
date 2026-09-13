#!/usr/bin/env bash

spatial_interactor_normalize_accidental_dollar_path() {
  local name="$1"
  local value="${!name:-}"

  if [[ "$value" == '$/'* ]]; then
    value="${value#\$}"
    printf -v "$name" '%s' "$value"
    export "$name"
    echo "Corrected ${name}: removed an accidental leading '$' -> ${value}" >&2
  fi
}

spatial_interactor_require_clean_absolute_path() {
  local name="$1"
  local value="$2"

  if [[ -z "$value" ]]; then
    echo "${name} must not be empty." >&2
    return 2
  fi
  if [[ "$value" != /* ]]; then
    echo "${name} must be an absolute path; got '${value}'. Remove any literal '$' before the leading slash." >&2
    return 2
  fi
  if [[ "$value" == *'$'* ]]; then
    echo "${name} contains an unexpanded '$': '${value}'. Use the resolved absolute path." >&2
    return 2
  fi
}
