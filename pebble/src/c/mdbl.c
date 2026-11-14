#include <pebble.h>

int main(void) {
  Window *w = window_create();
  window_stack_push(w, true);

  // Default XS heaps run out while building the 60-second tables; stack, slot and chunk
  // must all be non-zero or the record is rejected (coredevices/pebbleos#1621).
  ModdableCreationRecord cr = {
    .recordSize = sizeof(cr),
    .stack = 5 * 1024,
    .slot = 24 * 1024,
    .chunk = 24 * 1024,
#ifdef PBL_DEBUG
    // Built with `pebble build --debug`: enable the xsbug JavaScript debugger.
    .flags = kModdableCreationFlagDebug,
#endif
  };
  moddable_createMachine(&cr);

  window_destroy(w);
}
