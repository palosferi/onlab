# Szakdolgozat témavázlat

**Hallgató:** Pálos Ferenc (U1OTPN), BSc 7. félév, Intelligens hálózatok szakirány
**Konzulens:** Sonkoly Balázs
**Dátum:** 2026. szeptember 10.

**Munkacím:** A website fingerprinting támadások időbeli romlása a Tor hálózaton:
longitudinális mérés és a modellfrissítés költséghatékonysága

---

## 1. Kiindulópont

A 2025/26/2 félévben önálló laboratóriumot végeztem WF témában (konzulens:
Ladóczki Bence). Elkészült egy automatizált adatgyűjtő és kiértékelő lánc,
2237 PCAP-et rögzítettem 36 monitorozott és 20 ismeretlen webhelyről, és
összehasonlítottam egy Random Forestet egy Triplet MLP + KNN modellel zárt
világú, nyílt világú és zero-shot beállításban (baseline: 80,44%, obfs4:
73,33%, zero-shot: 15,08% pontosság). Az adathalmaz és a kód nyilvános.

Az önlab munkaterve eredetileg a concept drift vizsgálatát tűzte ki célul, de
erre a féléves keret nem adott lehetőséget: a beszámoló a jelenséget a
jövőbeli munkák között nevezi meg. A szakdolgozat ezt a kérdést viszi végig.

**Fontos korlát, amit tisztán kell látni:** a tavaszi adathalmaz nem
longitudinális. A baseline forgalom 2026. március 13-án és 26-27-én, az obfs4
április 3-4-én, az open-world adat április 22-én készült. Ez tehát egyetlen
időpont (t0), nem idősor. A drift-görbe minden további pontját mostantól kell
gyűjteni. Ez az egyetlen olyan eleme a tervnek, ami nem pótolható később.

---

## 2. Kutatási kérdések

**K1. Milyen ütemben romlik egy WF osztályozó pontossága, ahogy a tanító- és
tesztadat közötti időbeli távolság nő?**
A t0 és a december eleji utolsó kör között kb. 36 hét telik el, közben heti
felbontású mérési pontokkal. A romlás mértékét a drift nélküli felső korláthoz
(azonos körön belüli tanítás és tesztelés) mérem, mert enélkül nem
megkülönböztethető, hogy a modell romlott-e, vagy eleve gyenge volt.

**K2. Eltér-e a romlás üteme webhelytípusonként, és hogyan hat rá az obfs4?**
A céloldal-lista eleve három csoportra oszlik: statikus, hírportál, valamint
kereskedelmi és média oldalak. A hipotézis szerint a hírportálok gyorsabban
driftelnek. Az obfs4 réteg saját eloszlás-eltolódást ad hozzá; a kérdés, hogy
az időbeli és a protokoll-szintű eltolódás összeadódik-e vagy elfedi egymást.
Ez a tavaszi adatból elvileg sem volt megválaszolható, mert ott a protokoll és
az idő teljesen összekeveredett: minden baseline mérés március, minden obfs4
mérés április. Az új gyűjtés ezért mindkét ágat ugyanazon a körön belül,
váltogatva rögzíti.

**K3. Milyen újratanítási stratégia tartja a támadó pontosságát adott szinten
a legkisebb címkézési költséggel?**
Összehasonlítandó: nincs újratanítás, periodikus újratanítás 1, 2 és 4 hetente,
valamint csúszóablakos újratanítás. A költség mértékegysége a felhasznált
címkézett minták száma, mert a valóságban ez az, amit a támadónak elő kell
állítania.

---

## 3. Módszer és mérési elrendezés

- **Adatgyűjtés:** heti egy kör, fix napon és fix napszakban, 36 céloldal,
  körönként 5 ismétlés, két ágon (sima Tor és obfs4 híd). Kör = 360 mérés,
  kb. 3,7 óra. A látogatási sorrend a kör azonosítójából származtatott
  véletlen permutáció, így egy hálózati kiesés nem egyetlen webhely teljes
  körét viszi el.
- **Érvényesítés:** minden mérésnél rögzítem a végső URL-t, az oldalcímet, a
  DOM méretét és a betöltési időt, és mintaillesztéssel kiszűröm a Cloudflare
  és CAPTCHA interstitial oldalakat. Ezek nélkül egy blokkolt oldal tökéletes
  PCAP-et ad, amiben a céloldal forgalma nincs benne, és ez nem létező driftet
  gyárt. Az elutasított mérések nem törlődnek: a heti elutasítási arány maga is
  eredmény, mert azt mutatja, hogy egy oldal mikortól zárja ki a Tor exit node-okat.
- **Kovariánsok:** körönként rögzül a Tor verzió, a consensus időbélyege, a
  guard ujjlenyomata, a böngésző verziója és a mérőgép adatai. A drift
  magyarázatához ezek kellenek, különben egy Tor kiadás hatása
  megkülönböztethetetlen a webhelyek változásától.
- **Egy elkerülhetetlen eltérés a t0-hoz képest:** a tavaszi mérések mind a
  `th4r` guardon (27A06581, 57.129.38.230) mentek keresztül, ez a relay
  azonban időközben kikerült a consensusból. Az új sorozat ezért szükségszerűen
  másik guardot használ (`Sol`, 51.81.93.109), amit a mérés kezdetén rögzítek
  és a félév végéig változatlanul tartok. Ez valós, nem eltüntethető kovariáns:
  a t0 és az első új kör között a hálózati útvonal megváltozása és a webhelyek
  változása nem választható szét teljesen. A heti körök egymáshoz képest
  viszont már azonos útvonalon készülnek, így a drift-görbe meredekségét ez
  nem torzítja.
- **Az obfs4 ág saját, privát hídon fut.** A tavaszi obfs4 mérések hídja
  (109.110.170.208) időközben elérhetetlenné vált, és a bridges.torproject.org
  által kiadott öt csere-híd egyike sem válaszolt. A Tor Project saját
  hibajegye szerint a BridgeDB rendszeresen ad ki offline hidakat, mert a
  elérhetőségi adatai késnek. Egy 13 hetes sorozat nem építhető erre.
  Ezért egy saját, nem publikált obfs4 hidat üzemeltetek (Azure, Poland
  Central, `BridgeDistribution none`, `PublishServerDescriptor 0`), amelyhez
  csak a mérőgép csatlakozik. Így az obfs4 útvonal a félév végéig változatlan.
  A híd helye és Tor verziója kovariánsként rögzítendő.
- **Jellemzők:** a tavaszi kinyerő kód változatlan újrafelhasználásával, hogy
  egy szeptemberi jellemző ugyanazt jelentse, mint egy márciusi. A
  jellemzőkiválasztás kizárólag a t0-n történik és a teljes idősoron rögzített
  marad, különben a jövőbeli adat beszivárog a modellbe, ami éppen az „új
  adatot nem látó" stratégiát szépítené meg.
- **Modellek:** első körben a tavaszi Random Forest, hogy az eredmények
  visszavezethetők legyenek az önlabra. Októbertől egy erős, irodalmi
  referencia-támadás is (k-fingerprinting vagy Deep Fingerprinting a nyers
  irányszekvencián). Ez azért kell, mert a jelenlegi 21 aggregált jellemző az
  önlab beszámolójában is a legfőbb korlátként szerepel, és e nélkül a
  drift-eredmény azzal támadható, hogy nem a WF romlott, hanem a jellemzőkészlet
  volt gyenge. Újragyűjtést nem igényel: a csomagszintű CSV-k már megvannak.

---

## 3b. Az első mérési kör eredménye (2026-W37, szeptember 10.)

A mérőlánc működik: az első longitudinális kör mindkét ágon lefutott, oldalanként
öt ismétléssel, validált oldalbetöltésekkel. A tavaszi t0-hoz képest mért eltérés
látványos, de **nem nevezhető driftnek**, és ennek kimondása fontosabb, mint maga
a szám.

| Mérés | Érték |
|---|---|
| t0-n belüli felső korlát (nincs drift) | 79,26% |
| t0-modell a szeptemberi körön | 15,92% |
| macro-F1 ugyanott | 10,81% |
| átlagos KS-eltolódás 27 oldalon | 0,79 |

A modellfüggetlen KS-teszt szerint az eltolódás **csoportonként gyakorlatilag
azonos**: statikus 0,807, hírportál 0,791, kereskedelmi 0,773. Ha ez a webhelyek
tartalmi változásából eredne, a hírportáloknak érdemben jobban kellett volna
driftelniük a statikus oldalaknál. Az egyenletes eltolódás közös okra utal, ami
minden oldalt egyformán érint: a hálózati útvonal megváltozására. Ezt két ismert
tényező magyarázza:

1. **Guard-csere.** A t0 minden mérése a `th4r` guardon ment át, ez a relay
   azóta kikerült a consensusból, az új sorozat a `Sol` guardot használja. Az
   aggregált jellemzők nagy része (csomagszám, bájtszám, csomagközi idők) közvetlenül
   függ az útvonaltól.
2. **A t0 nem volt validálva.** A tavaszi gyűjtő nem ellenőrizte, hogy az oldal
   valóban betöltött-e, így a t0 nagy valószínűséggel tartalmaz Cloudflare
   challenge-oldalakat a céloldalak címkéjével. A szeptemberi kör viszont
   validált. A két halmaz tehát nem ugyanazt méri.

**Ebből következik a legfontosabb módszertani állítás:** a t0 és az első új kör
közötti lépés terhelt, a heti körök egymáshoz képest viszont azonos körülmények
között készülnek, így a drift-görbe **meredeksége** W37-től kezdve tiszta. A
dolgozat ezért a t0-t kiindulási referenciaként kezeli, nem a drift-görbe első
pontjaként.

**Javasolt kontrollkísérlet.** A guard-hatás egy olcsó méréssel elválasztható az
időbeli hatástól: ugyanazon a napon, ugyanazon a céloldal-listán egy második kör
egy *másik* guardon. Ha az így mért KS-eltolódás nagyságrendileg megegyezik a
t0-hoz mért 0,79-cel, az bizonyítja, hogy a jelenség útvonal-eredetű és nem
időbeli. Ez egy nap alatt elvégezhető.

### Oldalak, amelyek kizárják a Tort

A 36 céloldalból 9 nem adott használható mintát. Ez nem gyűjtési hiba, hanem
önálló eredmény, és a heti elutasítási arány mérhető mennyiség:

- **Cloudflare challenge:** w3c, stackoverflow, quora (,,Just a moment...''),
  medium (,,Attention Required''), reuters (,,Access Denied'')
- **HTTP 403:** imdb
- **Betöltési hiba:** cnn, gnu, vimeo

A zárt világú feladat így a gyakorlatban 36 helyett 27 osztályos. Mivel a
tavaszi gyűjtő nem validált, elképzelhető, hogy ezek egy része már márciusban is
blokkolt, csak akkor challenge-oldalként bekerült az adathalmazba.

## 4. Ütemterv

| Időszak | Feladat |
|---|---|
| szept. 8-13. (W37) | Mérőkörnyezet újraépítése, két Tor példány, előellenőrzés, 1. kör |
| szept. 14 - dec. 6. | Heti körök folyamatosan, összesen kb. 12 mérési pont |
| szeptember | Irodalom: Cherubin (USENIX 2022), Juarez, Rimmer, Sirinam, drift-irodalom |
| október | Erős referencia-támadás implementálása, futtatás a teljes eddigi idősoron |
| október vége | Első drift-görbe és köztes egyeztetés |
| november | Újratanítási stratégiák kiértékelése, költséggörbe, dolgozatírás |
| november vége | Nulladik verzió konzulensi véleményezésre |
| december eleje | Javítás, végleges leadás |

## 5. Kockázatok

| Kockázat | Kezelés |
|---|---|
| Kimarad egy heti kör | `Persistent=true` systemd timer, `--resume`, heti kétperces ellenőrzés |
| A guard rotálódik, üres PCAP-ek | A peer címe futásidőben a Tor vezérlőportjáról jön, üres mérés után újrafeloldás |
| Egy webhely tartósan blokkolja a Tort | Az érvényesítés kiszűri, az arány külön eredményként jelenik meg |
| A drift kicsi és nem szignifikáns | A KS-teszt alapú, modellfüggetlen eltolódás-mérés így is publikálható eredmény |
| Elromlik a mérőgép | A mérőgép cseréje kovariánsként rögzítendő, nem elhallgatandó |

## 6. Amiben a konzulens véleményét kérem

1. Elég-e a mérés önmagában, vagy legyen benne javaslat is: drift-detektor
   vagy újratanítási politika, amit a dolgozat kiértékel?
2. A Random Forest mellé melyik erős referencia-támadás legyen? Deep
   Fingerprinting vagy k-fingerprinting?
3. Maradjon a headless Chrome SOCKS proxyn, vagy váltsak valódi Tor Browserre?
   Az utóbbi hűbb a valósághoz, de elvágja az összehasonlíthatóságot a tavaszi
   t0-val. Elképzelhető egy kisebb párhuzamos Tor Browser ág is.
4. Jó-e a heti felbontás 36 oldalon, vagy inkább kevesebb oldal sűrűbben?
