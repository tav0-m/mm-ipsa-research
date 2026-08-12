"""Compatibilidad: use preferentemente `mm-ipsa run`."""

from mm_ipsa.pipeline import main

raise SystemExit(main())
