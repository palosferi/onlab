# Szakdolgozat feladatkiírás – tervezet

Konzulens: Dr. Sonkoly Balázs (TMIT)

## Cím

**Mélytanulás-alapú weboldal-ujjlenyomatozás általánosítása a Tor hálózaton**

## A feladat leírása

A Tor hálózat elrejti, hogy az interneten böngésző felhasználó melyik
weboldalt látogatja meg, a titkosított forgalom mérete, iránya és időzítése
azonban így is információt hordoz erről. Erre épül a weboldal-ujjlenyomatozás
(website fingerprinting): a felhasználó és a Tor hálózat közötti forgalmat
lehallgató fél gépi tanulással felismeri a meglátogatott oldalt. A legismertebb
mélytanulás-alapú változat a Deep Fingerprinting (DF) nevű konvolúciós neurális
háló, amely kísérleti környezetben nagy pontossággal azonosítja az oldalakat.
Az ilyen támadásokat jellemzően egyetlen, egységes körülmények között gyűjtött
adathalmazon mérik. Kevésbé ismert, mennyire működnek, ha a forgalom álcázott
kapcsolaton (pluggable transport, pl. obfs4 híd vagy Snowflake) jut be a Tor
hálózatba, ahogyan az a cenzúrázott hálózatokból csatlakozó felhasználóknál
jellemző.

A hallgató feladata annak vizsgálata, hogy a DF-támadás mennyire általánosít
három Tor-kapcsolódási mód (közvetlen kapcsolat, obfs4 híd és Snowflake)
között.

A hallgató feladatának a következőkre kell kiterjednie:

- A weboldal-ujjlenyomatozás és a Deep Fingerprinting szakirodalmának,
  valamint a Tor álcázott kapcsolódási módjainak áttekintése.
- Automatizált mérőkörnyezet kialakítása, amely a valódi Tor Browsert vezérelve
  a három kapcsolódási módon felváltva látogat meg weboldalakat, és rögzíti a
  keletkező hálózati forgalmat.
- Adathalmaz gyűjtése 50 kiválasztott (megfigyelt) és 500 további
  weboldalról, valamint olyan látogatásokról, amelyek közben egy második
  böngészőlapon egy másik oldal is töltődik.
- A DF-modell implementálása és kiértékelése: mennyire ismeri fel az oldalakat,
  ha ugyanazon vagy ha egy másik kapcsolódási mód forgalmán tanítják, és
  javít-e, ha a tanítóadat több módot is tartalmaz.
- A támadás vizsgálata nyílt világ esetén is, amikor a felhasználó a megfigyelt
  oldalakon kívül bármely oldalt meglátogathat.
- A kapcsolódási módok közötti teljesítménycsökkenés okainak elemzése a
  forgalom jellemzői alapján, valamint a párhuzamosan töltődő második oldal
  hatásának vizsgálata.
- Az eredmények értékelése, a módszer korlátainak és etikai szempontjainak
  bemutatása, továbbfejlesztési lehetőségek megfogalmazása.
