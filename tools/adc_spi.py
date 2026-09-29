#!/usr/bin/env python3
"""AD9643 SPI access over the FPGA's reg8/reg9 (design ID 0xAD964302).

  python3 tools/adc_spi.py selftest          chip ID + speed grade
  python3 tools/adc_spi.py dump              registers 0x00..0x3A, decoded
  python3 tools/adc_spi.py read  0x0D
  python3 tools/adc_spi.py write 0x0D 0x0F   (transfers automatically if the
                                              register is shadowed)
  python3 tools/adc_spi.py normal            force 0x0D = 0x00 + transfer

`write` refuses 0x09 bit0 (duty-cycle stabiliser) and 0x14 (output format):
the FPGA's timing analysis assumes the first, and the host's two's-complement
decoding assumes the second.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ad9643 as A
import diag

# Only the registers whose contents change how the rest of the system behaves.
DECODE = {
    0x01: lambda v: f"chip ID {'OK' if v == 0x82 else 'WRONG, expected 0x82'}",
    0x02: lambda v: f"speed grade bits5:4={(v>>4)&3:02b}"
                    f"{' (250 MSPS)' if ((v>>4)&3)==0 else ' NOT 250 MSPS'}",
    0x05: lambda v: f"channel enable A={'on' if v&1 else 'off'} "
                    f"B={'on' if v&2 else 'off'}",
    0x08: lambda v: f"power mode {v&3}" + (" (normal)" if (v&3)==0 else ""),
    0x09: lambda v: f"duty-cycle stabiliser {'ON' if v&1 else 'OFF  <-- MUST BE ON'}",
    0x0D: lambda v: {0x00:"normal ADC data", 0x0F:"RAMP test pattern",
                     0x04:"checkerboard", 0x05:"PN long",
                     0x06:"PN short"}.get(v & 0x0F, f"test mode 0x{v:02X}"),
    0x14: lambda v: "output format " + {1: "two's complement",
                                        0: "offset binary"}.get(v & 3, f"0b{v&3:02b}")
                    + ("" if (v & 3) == 1 else "  <-- host assumes two's complement"),
    0x17: lambda v: ("DCO delay OFF" if not (v & 0x80)
                     else f"DCO delay ON, {((v & 0x1F)+1)*100} ps"),
    0x18: lambda v: f"input span 0x{v:02X}",
}


def dump(adc, lo=0x00, hi=0x3A):
    print(f"  {'addr':>5} {'value':>6}   decode")
    for a in range(lo, hi + 1):
        v = adc.adc_rd(a)
        d = DECODE.get(a)
        note = d(v) if d else ""
        mark = " *" if a in A.ADC_SHADOWED else "  "
        print(f"  0x{a:02X}{mark} 0x{v:02X}   {note}")
    print("\n  * = shadowed: a write needs a transfer (0xFF <- 0x01) to take effect")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("dump")
    d.add_argument("--lo", type=lambda x: int(x, 0), default=0x00)
    d.add_argument("--hi", type=lambda x: int(x, 0), default=0x3A)
    r = sub.add_parser("read");  r.add_argument("addr", type=lambda x: int(x, 0))
    w = sub.add_parser("write")
    w.add_argument("addr", type=lambda x: int(x, 0))
    w.add_argument("data", type=lambda x: int(x, 0))
    w.add_argument("--force", action="store_true",
                   help="override the 0x09/0x14 guards (you had better mean it)")
    sub.add_parser("normal")
    a = p.parse_args()

    with A.Adc() as adc:
        try:
            adc.require_design_id("tools/adc_spi.py")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        if a.cmd == "selftest":
            r = diag.spi_selftest(adc)
            st = {"chip_id": r.data["chip_id"], "chip_id_ok": r.data["chip_id_ok"],
                  "grade_raw": r.data["grade_raw"],
                  "grade_bits": r.data["grade_bits"], "grade_ok": r.data["grade_ok"]}
            print(f"  chip ID     0x{st['chip_id']:02X}  "
                  f"{'PASS' if st['chip_id_ok'] else 'FAIL (expected 0x82)'}")
            print(f"  speed grade 0x{st['grade_raw']:02X}  bits5:4="
                  f"{st['grade_bits']:02b}  "
                  f"{'PASS (250 MSPS)' if st['grade_ok'] else 'FAIL'}")
            s = adc.adc_status()
            print(f"  reg10 0x{s['raw']:08X}  clk_locked={s['clk_locked']} "
                  f"ovr_A={s['overrange_a']} ovr_B={s['overrange_b']} "
                  f"idelay_ready={s['idelay_ready']}")
            return 0 if (st["chip_id_ok"] and st["grade_ok"]) else 1

        if a.cmd == "dump":
            dump(adc, a.lo, a.hi); return 0
        if a.cmd == "read":
            print(f"  0x{a.addr:02X} = 0x{adc.adc_rd(a.addr):02X}"); return 0
        if a.cmd == "write":
            adc.adc_wr_transfer(a.addr, a.data, force=a.force)
            print(f"  0x{a.addr:02X} <- 0x{a.data:02X}"
                  + ("  (+transfer)" if a.addr in A.ADC_SHADOWED else ""))
            print(f"  readback 0x{adc.adc_rd(a.addr):02X}")
            return 0
        if a.cmd == "normal":
            adc.adc_wr_transfer(A.ADC_TEST_MODE, A.ADC_TEST_NORMAL)
            print(f"  0x0D <- 0x00 (+transfer); readback "
                  f"0x{adc.adc_rd(A.ADC_TEST_MODE):02X}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
