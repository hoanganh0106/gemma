# E21: tile 512, eight partials
Parent: E17 candidate. Hypothesis: smaller tiles reduce live memory pressure.
Change: output projection uses eight 512-wide tiles and seven balanced partial additions, replacing four 1024-wide tiles and three additions.
Coverage: [0,512), [512,1024), [1024,1536), [1536,2048), [2048,2560), [2560,3072), [3072,3584), [3584,4096).
FP8 stream, weight-first DMA, Interleaved lane mode and other functions are unchanged from E17.
Additional BF16 rounding boundaries require official correctness validation.
Not compiled or submitted yet. before_e20.rs preserves the source previously on disk; parent_e17.rs is the comparison source.