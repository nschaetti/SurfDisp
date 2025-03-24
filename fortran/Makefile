FC = gfortran
SRC = surfdisp96_fixed.f
TARGET = libsurfdisp96.so
FFLAGS = -O3 -fPIC -shared -std=legacy -ffixed-form -ffixed-line-length-132

all: $(TARGET)

$(TARGET): $(SRC)
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
