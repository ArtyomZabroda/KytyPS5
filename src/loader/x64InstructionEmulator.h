#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX as a zero read and resumes after it.
void SkipLoad(void* native_context, size_t length);
// Completes a faulting 32-bit load into EAX with value, zero-extended, and resumes after it.
void CompleteLoad32(void* native_context, size_t length, uint32_t value);
// Reads a general-purpose register of a faulting context, numbered as in ModRM (0 is RAX).
[[nodiscard]] uint64_t ReadGpr(void* native_context, uint8_t index);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
