"""Compatibilidad: use preferentemente `mm-ipsa verify`."""

from mm_ipsa.verification import main

raise SystemExit(main())
