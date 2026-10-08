# Szakdolgozat feladatkiírás – tervezet

Konzulens: Dr. Sonkoly Balázs (TMIT)

## Cím

**Mélytanulás-alapú weboldal-ujjlenyomatozás általánosítása a Tor hálózaton**

*Generalization of Deep Learning-Based Website Fingerprinting on the Tor Network*

## A feladat leírása

A Tor hálózat célja, hogy elrejtse, melyik weboldalt keresi fel a felhasználó,
de a titkosított forgalom mérete és időzítése így is mintázatot hagy, amelyből
egy hálózati megfigyelő mélytanulással (pl. Deep Fingerprinting, DF) nagy
pontossággal visszaállíthatja a meglátogatott oldalt. A publikált támadásokat
jellemzően egyetlen, azonos körülmények között gyűjtött adathalmazon értékelik.
Kevésbé ismert, hogy egy ilyen modell mennyire marad hatékony, ha a forgalom
más Tor-átvitelen (pluggable transport) érkezik. Ez a cenzúrát megkerülő
felhasználók esetében gyakorlati kérdés.

A hallgató feladata a DF-támadás általánosíthatóságának vizsgálata három
átvitel között: sima Tor, obfs4 híd és Snowflake. A mérés valódi Tor Browser
forgalmon történik. A hallgató feladatai:

1. A témához kapcsolódó irodalom áttekintése (weboldal-ujjlenyomatozás,
   Deep Fingerprinting, Tor pluggable transportok, általánosítási problémák).
2. Automatizált adatgyűjtő környezet kialakítása és üzemeltetése: Tor Browser
   vezérlése Selenium-mal, a három átvitel váltogatva ugyanabban az ütemezésben,
   hálózati forgalom rögzítése és feldolgozása. Az adathalmaz 50 megfigyelt
   oldalt, 500 nem megfigyelt oldalt és egy háttérfüles (background tab) kísérletet
   tartalmaz.
3. A DF-modell implementálása és kiértékelése: zárt világ, átvitelek közötti
   (cross-transport) 3×3 mátrix, leave-one-out és közös (pooled) tanítás,
   valamint nyílt világ TPR/FPR értékekkel. A felosztás időrendi, az eredmények
   több futtatás (seed) átlaga és szórása.
4. Az átvitelek közötti teljesítménycsökkenés okainak elemzése (csomag- és
   cella-alapú reprezentáció, szekvenciahossz, átvitelenkénti forgalmi
   jellemzők), valamint a háttérfül hatásának vizsgálata.
5. Az eredmények értékelése, a módszer korlátainak és az etikai szempontoknak a
   bemutatása, továbbfejlesztési lehetőségek (pl. időbeli drift) megfogalmazása.

---

## Task description (English)

The Tor network hides which website a user visits, but the size and timing of
the encrypted traffic still leave patterns that a network observer can exploit
with deep learning (e.g. Deep Fingerprinting, DF). Published attacks are
usually evaluated on a single dataset collected under uniform conditions. How
well such a model transfers to traffic carried over a different Tor pluggable
transport is much less studied, although it matters for users in censored
networks.

The student investigates how well a DF attack generalizes across three
transports: plain Tor, obfs4 and Snowflake, using traffic from the real Tor
Browser. Tasks:

1. Review the literature on website fingerprinting, Deep Fingerprinting, Tor
   pluggable transports and generalization issues.
2. Build and operate an automated collection environment: Tor Browser driven by
   Selenium, the three transports interleaved in one schedule, traffic capture
   and processing. The dataset covers 50 monitored sites, 500 unmonitored sites
   and a background-tab experiment.
3. Implement and evaluate DF: closed world, a 3x3 cross-transport matrix,
   leave-one-out and pooled training, and open world with TPR/FPR. Splits are
   chronological; results are reported as mean and standard deviation over seeds.
4. Analyse the causes of the cross-transport drop (packet- vs cell-based
   representation, sequence length, per-transport traffic characteristics) and
   the effect of a background tab.
5. Evaluate the results, discuss limitations and ethical aspects, and outline
   future work (e.g. temporal drift).
