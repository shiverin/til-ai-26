# Shizhen Replay Collation

Analyzed 33 replay files from `replays`.
Score formula: `bases*10000 + buildings*300 + units*100 + hp + gold`.
Material score excludes banked gold.
Malformed lines skipped: 1.

## Consistent Top Scorers

| bot | top-score wins |
| --- | --- |
| yilin | 10 |
| gpt | 6 |
| v1 | 5 |
| baseline | 3 |
| random | 2 |
| xinyang | 2 |
| v1-defense | 1 |
| live-yilin | 1 |
| gemini | 1 |
| shizhen-gpt | 1 |
| gpt-diplomacy | 1 |

## Sole Survivors

| bot | sole-survivor wins |
| --- | --- |
| shizhen-gemini | 1 |
| gemini | 1 |

## Top-Gold Finishers

| bot | highest-gold finishes |
| --- | --- |
| yilin | 14 |
| v1 | 8 |
| random | 3 |
| gpt | 3 |
| xinyang | 2 |
| v1-gold-bank | 1 |
| live-yilin | 1 |
| shizhen-gpt | 1 |

## Family Summary

| family | entries | top-score wins | sole-survivor wins | top-gold finishes | avg score | avg gold | alive rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xinyang | 3 | 2 | 0 | 2 | 233308.7 | 172726.7 | 67% |
| yilin | 39 | 11 | 0 | 15 | 111589.8 | 75967.7 | 69% |
| gpt | 131 | 8 | 0 | 4 | 46049.7 | 16111.4 | 92% |
| baseline | 205 | 5 | 0 | 3 | 44687.5 | 41137.5 | 12% |
| v1 | 125 | 6 | 0 | 9 | 28207.4 | 4635.6 | 98% |
| gemini | 52 | 1 | 2 | 0 | 25236.3 | 5141.3 | 69% |

## Per-Replay Final Leaders

| match | turn | players | alive | top score | score | top gold | gold | top material | material | sole survivor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 011e99bd | 405 | 19 | 1 | random-4 | 282843 | random-4 | 280970 | shizhen-gemini | 67460 | shizhen-gemini |
| 144671d9 | 80 | 20 | 20 | xinyang | 199326 | xinyang | 29710 | xinyang | 169616 |  |
| 16760ce6 | 20 | 20 | 20 | gpt-09 | 16944 | v1-08 | 700 | gpt-09 | 16644 |  |
| 1756bc46 | 300 | 20 | 4 | yilin | 137950 | yilin | 121250 | shizhen-gemini | 67079 |  |
| 17d4da36 | 140 | 20 | 16 | v1-08 | 86549 | v1-09 | 39280 | target-gpt | 60638 |  |
| 1c96b390 | 5 | 20 | 20 | v1-defense | 12550 | v1-gold-bank | 250 | gpt-defense | 12400 |  |
| 1ff4e9f5 | 300 | 3 | 3 | yilin | 156324 | yilin | 98550 | shizhen-gpt | 65966 |  |
| 21ca51bf | 50 | 20 | 20 | gpt-03 | 58433 | gpt-09 | 6110 | gpt-03 | 53213 |  |
| 2b7b4e4c | 300 | 20 | 3 | xinyang | 488340 | xinyang | 488340 | yilin | 177190 |  |
| 3261f06f | 21 | 20 | 20 | v1-07 | 26041 | v1-01 | 910 | v1-07 | 25771 |  |
| 346609fd | 44 | 20 | 18 | live-yilin | 45712 | live-yilin | 820 | live-yilin | 44892 |  |
| 38bbab0f | 278 | 20 | 1 | gemini | 158349 | random-10 | 132630 | gemini | 67239 | gemini |
| 3d0370c1 | 140 | 20 | 20 | gpt-frozen-07 | 134044 | gpt-frozen-07 | 66630 | gpt-frozen-07 | 67414 |  |
| 51d00ac9 | 300 | 20 | 3 | yilin | 154690 | yilin | 154690 | shizhen-gemini | 66642 |  |
| 57044de6 | 20 | 20 | 20 | gpt-09 | 16944 | v1-08 | 700 | gpt-09 | 16644 |  |
| 7195dd2a | 2 | 2 | 2 | baseline | 11920 | yilin | 320 | baseline | 11900 |  |
| 7f696433 | 140 | 20 | 19 | yilin-09 | 131110 | yilin-09 | 72890 | gpt-air-control | 69026 |  |
| 83cff89e | 24 | 20 | 20 | v1-07 | 47371 | v1-01 | 1440 | v1-07 | 46981 |  |
| 9372b7e1 | 300 | 20 | 3 | yilin | 215626 | yilin | 151950 | yilin | 63676 |  |
| 96b52470 | 1 | 4 | 4 | baseline | 11910 | yilin | 410 | baseline | 11900 |  |
| 988f0aab | 30 | 3 | 3 | v1 | 16530 | v1 | 1430 | gemini | 15608 |  |
| 9caa2157 | 140 | 20 | 15 | v1-02 | 84983 | v1-02 | 36600 | gpt-frozen-02 | 64397 |  |
| ab844bde | 300 | 20 | 2 | shizhen-gpt | 185152 | yilin | 110150 | shizhen-gpt | 79672 |  |
| af863375 | 300 | 20 | 3 | yilin | 292898 | yilin | 151010 | yilin | 141888 |  |
| b7e63af5 | 3056 | 20 | 2 | gpt | 1352805 | gpt | 1280770 | gpt | 72035 |  |
| c2f5fc26 | 133 | 3 | 3 | yilin | 148610 | shizhen-gpt | 33370 | yilin | 116370 |  |
| c999a628 | 2 | 2 | 2 | baseline | 11920 | yilin | 320 | baseline | 11900 |  |
| cada2771 | 300 | 20 | 3 | yilin | 139000 | yilin | 139000 | shizhen-gemini | 78919 |  |
| d53c42af | 300 | 20 | 3 | yilin | 228112 | yilin | 173350 | shizhen-gemini | 60318 |  |
| dbed035f | 120 | 19 | 16 | random-12 | 93068 | random-12 | 69350 | random-3 | 24570 |  |
| ddd6fe21 | 20 | 20 | 20 | gpt-09 | 16934 | v1-08 | 700 | gpt-09 | 16634 |  |
| f276a8bb | 300 | 20 | 3 | yilin | 327651 | yilin | 205470 | yilin | 122181 |  |
| f9d8bde1 | 23 | 20 | 20 | gpt-diplomacy | 38118 | yilin | 530 | gpt-diplomacy | 37738 |  |

## Highest End-State Gold

| rank | match | bot | gold | score | material | bases | buildings | units | hp | alive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | b7e63af5 | gpt | 1280770 | 1352805 | 72035 | 4 | 25 | 72 | 17335 | True |
| 2 | 2b7b4e4c | xinyang | 488340 | 488340 | 0 | 0 | 0 | 0 | 0 | False |
| 3 | b7e63af5 | yilin-02 | 322740 | 343778 | 21038 | 1 | 6 | 44 | 4838 | True |
| 4 | 011e99bd | random-4 | 280970 | 282843 | 1873 | 0 | 5 | 0 | 373 | False |
| 5 | 011e99bd | random-3 | 246770 | 246770 | 0 | 0 | 0 | 0 | 0 | False |
| 6 | 011e99bd | random-16 | 226690 | 226690 | 0 | 0 | 0 | 0 | 0 | False |
| 7 | 011e99bd | random-12 | 215310 | 215310 | 0 | 0 | 0 | 0 | 0 | False |
| 8 | 011e99bd | random-17 | 209170 | 209170 | 0 | 0 | 0 | 0 | 0 | False |
| 9 | f276a8bb | yilin | 205470 | 327651 | 122181 | 7 | 50 | 146 | 22581 | True |
| 10 | 011e99bd | random-13 | 197410 | 197410 | 0 | 0 | 0 | 0 | 0 | False |
| 11 | 011e99bd | random-2 | 179930 | 179930 | 0 | 0 | 0 | 0 | 0 | False |
| 12 | 011e99bd | random-7 | 174490 | 174490 | 0 | 0 | 0 | 0 | 0 | False |
| 13 | d53c42af | yilin | 173350 | 228112 | 54762 | 2 | 27 | 104 | 16262 | True |
| 14 | 011e99bd | random-9 | 166860 | 166860 | 0 | 0 | 0 | 0 | 0 | False |
| 15 | 2b7b4e4c | yilin | 162330 | 339520 | 177190 | 8 | 65 | 340 | 43690 | True |

## Highest End-State Score

| rank | match | bot | score | gold | material | bases | buildings | units | hp | alive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | b7e63af5 | gpt | 1352805 | 1280770 | 72035 | 4 | 25 | 72 | 17335 | True |
| 2 | 2b7b4e4c | xinyang | 488340 | 488340 | 0 | 0 | 0 | 0 | 0 | False |
| 3 | b7e63af5 | yilin-02 | 343778 | 322740 | 21038 | 1 | 6 | 44 | 4838 | True |
| 4 | 2b7b4e4c | yilin | 339520 | 162330 | 177190 | 8 | 65 | 340 | 43690 | True |
| 5 | f276a8bb | yilin | 327651 | 205470 | 122181 | 7 | 50 | 146 | 22581 | True |
| 6 | af863375 | yilin | 292898 | 151010 | 141888 | 8 | 48 | 186 | 28888 | True |
| 7 | 011e99bd | random-4 | 282843 | 280970 | 1873 | 0 | 5 | 0 | 373 | False |
| 8 | 011e99bd | random-3 | 246770 | 246770 | 0 | 0 | 0 | 0 | 0 | False |
| 9 | d53c42af | yilin | 228112 | 173350 | 54762 | 2 | 27 | 104 | 16262 | True |
| 10 | 011e99bd | random-16 | 226690 | 226690 | 0 | 0 | 0 | 0 | 0 | False |
| 11 | 9372b7e1 | yilin | 215626 | 151950 | 63676 | 3 | 33 | 95 | 14276 | True |
| 12 | 011e99bd | random-12 | 215310 | 215310 | 0 | 0 | 0 | 0 | 0 | False |
| 13 | 011e99bd | random-17 | 209170 | 209170 | 0 | 0 | 0 | 0 | 0 | False |
| 14 | 144671d9 | xinyang | 199326 | 29710 | 169616 | 15 | 36 | 12 | 7616 | True |
| 15 | 011e99bd | random-13 | 197410 | 197410 | 0 | 0 | 0 | 0 | 0 | False |

## Highest End-State Material

| rank | match | bot | material | score | gold | bases | buildings | units | hp | alive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2b7b4e4c | yilin | 177190 | 339520 | 162330 | 8 | 65 | 340 | 43690 | True |
| 2 | 144671d9 | xinyang | 169616 | 199326 | 29710 | 15 | 36 | 12 | 7616 | True |
| 3 | af863375 | yilin | 141888 | 292898 | 151010 | 8 | 48 | 186 | 28888 | True |
| 4 | f276a8bb | yilin | 122181 | 327651 | 205470 | 7 | 50 | 146 | 22581 | True |
| 5 | c2f5fc26 | yilin | 116370 | 148610 | 32240 | 7 | 53 | 103 | 20170 | True |
| 6 | ab844bde | shizhen-gpt | 79672 | 185152 | 105480 | 5 | 25 | 73 | 14872 | True |
| 7 | cada2771 | shizhen-gemini | 78919 | 80089 | 1170 | 5 | 25 | 63 | 15119 | True |
| 8 | b7e63af5 | gpt | 72035 | 1352805 | 1280770 | 4 | 25 | 72 | 17335 | True |
| 9 | f276a8bb | shizhen-gpt | 69027 | 84237 | 15210 | 4 | 28 | 66 | 14027 | True |
| 10 | 7f696433 | gpt-air-control | 69026 | 85006 | 15980 | 5 | 25 | 30 | 8526 | True |
| 11 | 2b7b4e4c | shizhen-gpt | 67940 | 68910 | 970 | 4 | 25 | 67 | 13740 | True |
| 12 | 011e99bd | shizhen-gemini | 67460 | 178690 | 111230 | 4 | 26 | 72 | 12460 | True |
| 13 | 3d0370c1 | gpt-frozen-07 | 67414 | 134044 | 66630 | 5 | 24 | 25 | 7714 | True |
| 14 | 38bbab0f | gemini | 67239 | 158349 | 91110 | 4 | 26 | 59 | 13539 | True |
| 15 | 1756bc46 | shizhen-gemini | 67079 | 91769 | 24690 | 4 | 25 | 62 | 13379 | True |

## Bot Aggregate By Name

| bot | entries | avg score | avg gold | max score | max gold | max material | alive rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xinyang | 3 | 233308.7 | 172726.7 | 488340 | 488340 | 169616 | 67% |
| yilin | 36 | 118592.4 | 82201.7 | 343778 | 322740 | 177190 | 67% |
| shizhen-gpt | 14 | 64560.4 | 17392.1 | 185152 | 105480 | 79672 | 86% |
| shizhen-gemini | 14 | 49854.9 | 12220.7 | 178690 | 111230 | 78919 | 71% |
| gpt-diplomacy | 4 | 47701.8 | 9987.5 | 88038 | 33190 | 54848 | 100% |
| gpt-expansion-fortress | 4 | 46406 | 8335 | 81108 | 25150 | 56238 | 100% |
| baseline | 12 | 46183.2 | 10957.5 | 89078 | 39740 | 51964 | 92% |
| live-yilin | 1 | 45712 | 820 | 45712 | 820 | 44892 | 100% |
| gpt | 84 | 45323.4 | 19886.8 | 1352805 | 1280770 | 72035 | 92% |
| random | 193 | 44594.5 | 43014.0 | 282843 | 280970 | 24570 | 7% |
| gpt-scout-vision | 4 | 43292.2 | 7560 | 80159 | 21940 | 58219 | 100% |
| gpt-ground-rush | 2 | 41006.5 | 5675 | 70033 | 11270 | 58763 | 100% |
| gpt-balanced | 4 | 40480.2 | 3355 | 71488 | 6870 | 64618 | 100% |
| gpt-resource | 2 | 40030 | 5485 | 68010 | 10820 | 57190 | 100% |
| target-gpt | 4 | 38794 | 8302.5 | 78418 | 31970 | 60638 | 75% |
| gpt-defense | 2 | 37950 | 2640 | 63410 | 5190 | 58220 | 100% |
| gpt-air-control | 4 | 36371.8 | 4100 | 85006 | 15980 | 69026 | 100% |
| gpt-base-assault | 2 | 35665.5 | 1220 | 59281 | 2290 | 56991 | 100% |
| v1 | 92 | 29666.3 | 6030 | 87668 | 43610 | 49580 | 97% |
| v1-base-assault | 24 | 28642.2 | 982.5 | 51635 | 6750 | 44885 | 100% |
| yilin-current | 1 | 19936 | 1870 | 19936 | 1870 | 18066 | 100% |
| frozen-yilin | 1 | 17027 | 790 | 17027 | 790 | 16237 | 100% |
| gemini | 38 | 16166.3 | 2533.2 | 158349 | 91110 | 67239 | 68% |
| v1-defense | 1 | 12550 | 150 | 12550 | 150 | 12400 | 100% |
| v1-scout-vision | 1 | 12490 | 90 | 12490 | 90 | 12400 | 100% |
| v1-gold-bank | 1 | 12150 | 250 | 12150 | 250 | 11900 | 100% |
| v1-diplomacy | 1 | 12050 | 150 | 12050 | 150 | 11900 | 100% |
| v1-infantry-swarm | 1 | 12050 | 150 | 12050 | 150 | 11900 | 100% |
| v1-resource | 1 | 11980 | 80 | 11980 | 80 | 11900 | 100% |
| v1-ground-rush | 1 | 11980 | 80 | 11980 | 80 | 11900 | 100% |
| v1-air-control | 1 | 11980 | 80 | 11980 | 80 | 11900 | 100% |
| v1-balanced | 1 | 11980 | 80 | 11980 | 80 | 11900 | 100% |
| gpt-original | 1 | 10 | 10 | 10 | 10 | 0 | 0% |
