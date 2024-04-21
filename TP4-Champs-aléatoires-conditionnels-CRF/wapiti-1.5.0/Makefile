CFLAGS =-std=c99 -W -Wall -Wextra -O3
LIBS   =-lm -lpthread

DESTDIR=
PREFIX="/home/nas-wks01/users/uapv2403399/Donnees_itinerantes_depuis_serveur_pedagogique/Mes Documents/S2/Modeles_Stochastiques 2/TP4-Champs-aléatoires-conditionnels-CRF/wapiti-1.5.0"

INSTALL= install -p
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644

SRC=src/*.c
HDR=src/*.h

wapiti: $(SRC) $(HDR)
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -DNDEBUG $(CFLAGS) -o wapiti $(SRC) $(LIBS)

debug: $(SRC) $(HDR)
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -g $(CFLAGS) -o wapiti $(SRC) $(LIBS)

install: wapiti
	@echo "CP: wapiti   --> $(DESTDIR)$(PREFIX)/bin"
	@mkdir -p $(DESTDIR)$(PREFIX)/bin
	@mkdir -p $(DESTDIR)$(PREFIX)/share/man/man1
	@$(INSTALL_EXEC) wapiti       $(DESTDIR)$(PREFIX)/bin
	@$(INSTALL_DATA) doc/wapiti.1 $(DESTDIR)$(PREFIX)/share/man/man1

clean:
	@echo "RM: wapiti"
	@rm -f wapiti

.PHONY: clean install
