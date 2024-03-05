ifeq ($(findstring dfx, $(PLATFORM)), dfx)
CXXFLAGS += -Ddfx_device
endif
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(findstring vck190_base_dfx, $(PLATFORM)), vck190_base_dfx)
$(error [ERROR]: This example is not supported for sw_emu & hw_emu when targeting $(PLATFORM).)
endif
endif
