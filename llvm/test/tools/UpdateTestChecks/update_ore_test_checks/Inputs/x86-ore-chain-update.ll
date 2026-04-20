; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s --check-prefix ASM
; RUN: llc -mtriple=x86_64-linux-gnu -pass-remarks-output=- -pass-remarks-filter=asm-printer %s -o /dev/null | FileCheck %s --check-prefix REMARKS

define i32 @add(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}
