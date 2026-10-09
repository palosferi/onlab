# Szakdolgozat feladatkiírás – tervezet

Konzulens: Dr. Sonkoly Balázs (TMIT)

## Cím

**Mélytanulás-alapú weboldal-ujjlenyomatozás általánosítása a Tor hálózaton**

*Generalization of Deep Learning-Based Website Fingerprinting on the Tor Network*

## A feladat leírása

A Tor hálózat elrejti, hogy a felhasználó melyik weboldalt látogatja meg, de a
titkosított forgalom mérete, iránya és időzítése így is árulkodó. Erre épül a
weboldal-ujjlenyomatozás (website fingerprinting): a felhasználó és a Tor
hálózat közötti forgalmat lehallgató fél gépi tanulással felismeri a
meglátogatott oldalt. A legismertebb mélytanulás-alapú változat, a Deep
Fingerprinting (DF) konvolúciós neurális háló, amely kísérleti környezetben
98%-os pontosságot ért el. Az ilyen támadásokat jellemzően egyetlen, egységes
körülmények között gyűjtött adathalmazon mérik. Kevésbé ismert, mennyire
működnek, ha a forgalom álcázott kapcsolaton (pluggable transport, pl. obfs4
híd vagy Snowflake) jut be a Tor hálózatba; ezeket a cenzúrázott hálózatokból
csatlakozó felhasználók használják.

A hallgató feladata annak vizsgálata, hogy a DF-támadás mennyire általánosít
három kapcsolódási mód között: közvetlen Tor-kapcsolat, obfs4 híd és
Snowflake. A hallgató feladatai:

1. A szakirodalom áttekintése: weboldal-ujjlenyomatozás, Deep Fingerprinting,
   a Tor álcázott kapcsolódási módjai.
2. Automatizált mérőkörnyezet kialakítása, amely a valódi Tor Browsert vezérelve
   látogat meg weboldalakat a három kapcsolódási módon felváltva, és rögzíti a
   keletkező hálózati forgalmat. A mérés 50 kiválasztott (megfigyelt) weboldalt
   és 500 további népszerű oldalt fed le, valamint olyan látogatásokat, amelyek
   közben egy második böngészőlapon egy másik oldal is töltődik.
3. A DF-modell implementálása és kiértékelése: mennyire ismeri fel az oldalakat
   ugyanazon a kapcsolódási módon, illetve egy másik mód forgalmán tanítva;
   javít-e, ha a tanítóadat több módot is tartalmaz; és mennyire működik akkor,
   ha a felhasználó a megfigyelt oldalakon kívül bármit meglátogathat.
4. A kapcsolódási módok közötti teljesítménycsökkenés okainak elemzése a
   forgalom jellemzői alapján, valamint a párhuzamosan töltődő második oldal
   hatásának vizsgálata.
5. Az eredmények értékelése, a módszer korlátainak és etikai szempontjainak
   bemutatása, továbbfejlesztési lehetőségek megfogalmazása.

---

## Task description (English)

Tor hides which website a user visits, but the size, direction and timing of
the encrypted traffic still give it away. Website fingerprinting exploits this:
an observer between the user and the Tor network recognizes the visited page
with machine learning. The best-known deep-learning variant, Deep
Fingerprinting (DF), is a convolutional neural network that reached 98%
accuracy in a lab setting. Such attacks are usually evaluated on one dataset
collected under uniform conditions. Much less is known about how well they
work when traffic enters Tor through a disguised connection (a pluggable
transport such as an obfs4 bridge or Snowflake), which is what users in
censored networks rely on.

The student investigates how well DF generalizes across three ways of
connecting: direct Tor, an obfs4 bridge and Snowflake. Tasks:

1. Review the literature on website fingerprinting, Deep Fingerprinting and
   Tor's disguised connection methods.
2. Build an automated measurement setup that drives the real Tor Browser to
   visit websites over the three connection methods in turn and records the
   resulting traffic. It covers 50 selected (monitored) websites, 500 further
   popular sites, and visits during which a second site loads in another tab.
3. Implement and evaluate DF: how well it recognizes pages on the same
   connection method and when trained on another one; whether training on
   several methods helps; and how well it works when the user may visit any
   site, not only the monitored ones.
4. Analyse why accuracy drops between connection methods, based on traffic
   characteristics, and the effect of a second page loading in parallel.
5. Evaluate the results, discuss limitations and ethical aspects, and outline
   future work.
