"""
notebook_packaging
==================
Executed-notebook archives for readers (PR 42, Step 6B).

``accepted_executions.json`` lists every tracked notebook and, for each, its accepted
executions. Each execution has its run id, where its executed copy is read from and the
tracked record that accepts it. ``python -m notebook_packaging`` builds two kinds of archive
from that list, copying the executed files unchanged and never modifying them:

- ``executed_notebooks.zip``: the current accepted copy of each notebook, at the ZIP root, with its repository
  path recorded in the manifest (refreshed whenever a new execution is accepted);
- ``<pr>_executed_notebooks.zip``: one PR's accepted executions, with run ids in the file
  names (a fixed snapshot).

It also verifies archives, refusing any archive with an unsafe or colliding member path
before extracting it. And it clears tracked notebooks' outputs as a separate, explicit command
(``clear-outputs``). That command refuses unless a verified archive accounts for each notebook,
and it first preserves each notebook's current outputs outside the repository. ``register``
adds a newly accepted run to the manifest, and ``exclude`` records a snapshot member whose
executed copy is not available; both write only an accepted, validated manifest. See
``packaging`` for the commands and ``documentation/notebook_map.md`` section 6.
"""

from .packaging import (ARCHIVE_FORMAT, MANIFEST_FORMAT, PROJECT_ARCHIVE, PackagingError, build_project,
                        build_snapshot, clear_outputs, content_without_outputs, exclude_execution, inventory,
                        load_manifest, main, notebook_summary, register_execution, verify_archive)

__all__ = ["ARCHIVE_FORMAT", "MANIFEST_FORMAT", "PROJECT_ARCHIVE", "PackagingError", "build_project",
           "build_snapshot", "clear_outputs", "content_without_outputs", "exclude_execution", "inventory",
           "load_manifest", "main", "notebook_summary", "register_execution", "verify_archive"]
