# 20 份黄金理解(grok-4-20-reasoning, 8 段式)+ 原始问题 + v3 自由组合判决

## 1. `nt-1` — ❌ v3未救 | 模板: ordered_navigation
**问题**: in what city did piotr's last 1st place finish occur?
**金标**: Bangkok, Thailand
**表**: csv/204-csv/622.csv

```
1. Answer Type: value
2. Query Template: ordered_navigation
3. Explicit Atoms:
- piotr = piotr's
- finish = 1st place
- order = last
4. Reverse Tree: answer <- city
answer <- last row
row <- filtered set
5. Procedure: filter piotr's
filter 1st place finish
navigate ordered
select city
6. Targets: none
7. Order Semantics: last = final in sequence of piotr's 1st place finishes
8. Traps: none
```

## 2. `nt-3` — ✅ v3拯救 | 模板: comparison
**问题**: how many more passengers flew to los angeles than to saskatoon from manzanillo airport in 2013?
**金标**: 12,467
**表**: csv/203-csv/515.csv

```
1. Answer Type: value
2. Query Template: comparison
3. Explicit Atoms:
- flew to = los angeles
- to = saskatoon
- from = manzanillo airport
- in = 2013
4. Reverse Tree: answer <- LA_count - SK_count
LA_count, SK_count <- filter(manzanillo airport, 2013)
5. Procedure: filter manzanillo airport
filter 2013
aggregate passengers to los angeles
aggregate passengers to saskatoon
subtract
6. Targets: passengers to los angeles from manzanillo airport in 2013
passengers to saskatoon from manzanillo airport in 2013
7. Order Semantics: none
8. Traps: none
```

## 3. `nt-4` — ✅ v3拯救 | 模板: ordered_navigation
**问题**: who was the opponent in the first game of the season?
**金标**: Derby County
**表**: csv/204-csv/495.csv

```
1. Answer Type: value
2. Query Template: ordered_navigation
3. Explicit Atoms:
- game = first
- season
4. Reverse Tree: answer <- opponent
answer <- first game row
5. Procedure: filter season
order by sequence
select first
extract opponent
6. Targets: none
7. Order Semantics: first means initial entry over season sequence
8. Traps: none
```

## 4. `nt-5` — ❌ v3未救 | 模板: filter_aggregate
**问题**: how many people stayed at least 3 years in office?
**金标**: 4
**表**: csv/203-csv/705.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- stayed = at least 3 years in office
4. Reverse Tree: answer <- count
count <- filtered set
5. Procedure:
filter people by stayed at least 3 years in office
count filtered people
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 5. `nt-6` — ❌ v3未救 | 模板: ordered_navigation
**问题**: who is the first away team on the chart
**金标**: Varbergs GIF
**表**: csv/204-csv/361.csv

```
1. Answer Type: value
2. Query Template: ordered_navigation
3. Explicit Atoms:
- position = first
- qualifier = away team
4. Reverse Tree: answer <- first matching entry
<- chart
5. Procedure:
navigate chart from start
filter qualifier away team
select first entry
6. Targets: none
7. first means earliest listed position over chart entries
8. Traps: none
```

## 6. `nt-7` — ❌ v3未救 | 模板: comparison
**问题**: which is deeper, lake tuz or lake palas tuzla?
**金标**: Lake Palas Tuzla
**表**: csv/204-csv/341.csv

```
1. Answer Type: value
2. Query Template: comparison
3. Explicit Atoms:
- lake = tuz
- lake = palas tuzla
4. Reverse Tree: answer <- deeper lake
answer <- depth(lake tuz) vs depth(lake palas tuzla)
5. Procedure: lookup depth for lake tuz
lookup depth for lake palas tuzla
compare the two depths
select lake with larger depth
6. Targets: depth for lake tuz
depth for lake palas tuzla
7. Order Semantics: none
8. Traps: none
```

## 7. `nt-9` — ❌ v3未救 | 模板: comparison
**问题**: which players played the same position as ardo kreek?
**金标**: Siim Ennemuist|Andri Aganits
**表**: csv/203-csv/116.csv

```
1. Answer Type: list
2. Query Template: comparison
3. Explicit Atoms:
- same position as = ardo kreek
4. Reverse Tree: answer <- matching players
- matching players <- position of ardo kreek
5. Procedure:
- lookup position of ardo kreek
- filter players matching position
- select players
6. Targets:
position of ardo kreek
7. Order Semantics: none
8. Traps: none
```

## 8. `nt-10` — ❌ v3未救 | 模板: filter_aggregate
**问题**: how many times did an italian cyclist win a round?
**金标**: 6
**表**: csv/204-csv/253.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- cyclist = italian
4. Reverse Tree:
answer <- count
count <- filtered set
5. Procedure:
filter cyclist = italian
count rounds won
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 9. `nt-12` — ❌ v3未救 | 模板: comparison
**问题**: what is the difference in the number of temples between imabari and matsuyama?
**金标**: 2
**表**: csv/204-csv/841.csv

```
1. Answer Type: value
2. Query Template: comparison
3. Explicit Atoms:
- imabari
- matsuyama
4. Reverse Tree: answer <- difference
difference <- number of temples (imabari, matsuyama)
5. Procedure: filter imabari
retrieve number of temples
filter matsuyama
retrieve number of temples
compute difference
6. Targets: number of temples for imabari
number of temples for matsuyama
7. Order Semantics: none
8. Traps: none
```

## 10. `nt-14` — ❌ v3未救 | 模板: row_lookup
**问题**: which athlete was from south korea after the year 2010?
**金标**: Kim Yu-na
**表**: csv/203-csv/104.csv

```
1. Answer Type: list
2. Query Template: row_lookup
3. Explicit Atoms:
- from = south korea
- after the year = 2010
4. Reverse Tree: answer <- filtered set
5. Procedure:
filter after the year 2010
filter from south korea
select athlete
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 11. `nt-18` — ❌ v3未救 | 模板: filter_aggregate
**问题**: how many movies have links to other wikipedia pages about them?
**金标**: 20
**表**: csv/201-csv/21.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- movies = have links to other wikipedia pages about them
4. Reverse Tree: answer <- count <- filtered set
5. Procedure: filter movies
count rows
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 12. `nt-20` — ❌ v3未救 | 模板: grouped_extreme
**问题**: which album released by the band schnell fenster produced the most singles appearing on the australian peak chart?
**金标**: The Sound Of Trees
**表**: csv/202-csv/184.csv

```
1. Answer Type: value
2. Query Template: grouped_extreme
3. Explicit Atoms:
- band = schnell fenster
- chart = australian peak chart
4. Reverse Tree: answer <- album
answer <- max count per album
answer <- filtered schnell fenster singles
5. Procedure: filter band
filter chart appearance
group by album
count per group
select max
6. Targets: singles-per-album counts
7. Order Semantics: most = highest count of singles on chart
8. Traps: none
```

## 13. `nt-23` — ❌ v3未救 | 模板: grouped_extreme
**问题**: what counties had the least participants for the race?
**金标**: Morocco|France|Spain
**表**: csv/204-csv/849.csv

```
1. Answer Type: list
2. Query Template: grouped_extreme
3. Explicit Atoms:
- participants = least
4. Reverse Tree: answer <- extreme counties <- participant counts
5. Procedure:
group by counties
count participants
identify least count
select matching counties
6. Targets: participant counts per county
7. Order Semantics: none
8. Traps: ties
```

## 14. `nt-24` — ❌ v3未救 | 模板: ordered_navigation
**问题**: who ranked right after turkey?
**金标**: Sweden
**表**: csv/203-csv/812.csv

```
1. Answer Type: value
2. Query Template: ordered_navigation
3. Explicit Atoms:
- right after = turkey
4. Reverse Tree: answer <- next after turkey <- ranking order
5. Procedure: locate turkey
identify its position
navigate to next position
select corresponding entry
6. Targets: none
7. Order Semantics: right after means immediate successor in ranking
8. Traps: none
```

## 15. `nt-25` — ✅ v3拯救 | 模板: filter_aggregate
**问题**: what's the number of parishes founded in the 1800s?
**金标**: 4
**表**: csv/203-csv/36.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- founded = 1800s
4. Reverse Tree: answer <- filtered set
5. Procedure:
filter founded = 1800s
count
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 16. `nt-27` — ❌ v3未救 | 模板: ordered_navigation
**问题**: in 1996 the sc house of representatives had a republican majority. how many years had passed since the last time this happened?
**金标**: 122
**表**: csv/203-csv/95.csv

```
1. Answer Type: value
2. Query Template: ordered_navigation
3. Explicit Atoms:
- sc house of representatives = republican majority
- 1996
- last time this happened
4. Reverse Tree: answer <- year difference
answer <- 1996 - prior year
prior year <- filtered set
5. Procedure: filter sc house of representatives with republican majority
order by year
locate 1996
select prior entry
subtract years
6. Targets: none
7. Order Semantics: last = most recent prior to 1996 over years
8. Traps: none
```

## 17. `nt-30` — ❌ v3未救 | 模板: filter_aggregate
**问题**: what was the average number of years served by a coach?
**金标**: 4
**表**: csv/203-csv/577.csv

```
1. Answer Type: value
2. Query Template: filter_aggregate
3. Explicit Atoms:
4. Reverse Tree: answer <- average
<- number of years served by a coach
5. Procedure: aggregate average
of number of years served by a coach
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 18. `nt-31` — ❌ v3未救 | 模板: filter_aggregate
**问题**: how many beta versions were released before the first full release?
**金标**: 9
**表**: csv/203-csv/743.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- versions = beta
- release = full
- first full release
4. Reverse Tree: answer <- count
answer <- betas before first full release
first full <- earliest full
5. Procedure:
locate first full release
extract its date
filter beta versions released before
count filtered rows
6. Targets:
date of first full release
7. Order Semantics: first = earliest by release date over full releases
8. Traps: none
```

## 19. `nt-37` — ❌ v3未救 | 模板: filter_aggregate
**问题**: what is the total amount of processors that have both an integrated color display and an enhanced keyboard?
**金标**: 4
**表**: csv/203-csv/136.csv

```
1. Answer Type: count
2. Query Template: filter_aggregate
3. Explicit Atoms:
- have = integrated color display
- have = enhanced keyboard
4. Reverse Tree:
answer <- count
count <- filtered set
5. Procedure:
filter rows matching both conditions
count qualifying rows
6. Targets: none
7. Order Semantics: none
8. Traps: none
```

## 20. `nt-38` — ❌ v3未救 | 模板: ordered_navigation
**问题**: how many consecutive friendly competitions did chalupny score in?
**金标**: 2
**表**: csv/204-csv/920.csv

```
1. Answer Type: count
2. Query Template: ordered_navigation
3. Explicit Atoms:
- player = chalupny
- competition = friendly competitions
- action = score
4. Reverse Tree: answer <- streak length <- filtered set
5. Procedure:
filter friendly competitions
filter chalupny scored
order by match sequence
identify successive streaks
count sequence length
6. Targets: none
7. Order Semantics: consecutive means successive without gaps in ordered friendly competitions
8. Traps: none
```
