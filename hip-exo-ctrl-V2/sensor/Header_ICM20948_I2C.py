import time, traceback, copy
import numpy as np
# "pip install smbus" (rather than python-smbus, smbus2 or 3, etc.)
# ti.com says "Generally [the] SMBus [protocol] is compatible with I2C devices..."
import smbus

class ICM20948_I2C_IMUs:
    def __init__(self):
        # uncomment the following line and comment the line after it
        # to sweep all mux ports to see what's attached
        # self.IMU_mux_ports = dict([(f"{i}",i) for i in range(0,4)])
        self.IMU_mux_ports = {    # Multiplexer port numbers with an IMU actually attached
            'IMU_PELVIS' : 2, # correct pairing as of 20250114
            'IMU_THIGH_RIGHT' : 1, # correct pairing as of 20250114
            'IMU_THIGH_LEFT' : 4, # correct pairing as of 20250114
        }
        self.imu_readings = copy.deepcopy(self.IMU_mux_ports) # need same keys so just copy

        self.JETSON_I2C_BUS = 7         # I2C bus of Orin to which multiplexer is attached.
        self.IMU_ADDRESS = 0x69         # I2C/SMBus address of all IMUs on exoskeleton.
        self.IMU_ADDRESS_soldered = 0x68 # soldered: 0x68, unsoldered: 0x69
        if self.IMU_ADDRESS == self.IMU_ADDRESS_soldered:   print("\n !!!!! Warning: Trunk IMU is used. Check if correct IMU address. !!!!!\n")
        self.WHO_AM_I = 0x00            # I2C/SMBus address of WHO_AM_I register.
        self.YOU_ARE = 0xEA             # Value that should be stored in WHO_AM_I register.
        self.MULTIPLEXER_ADDRESS = 0x70 # I2C/SMBus address of multiplexer on exoskeleton.
        # TO DO: I thought I2C was single-device, but can we get rid of the mux?
        # https://learn.adafruit.com/i2c-addresses/overview

        self.BANK_SEL = 0x7F
        self.GYRO_CONFIG_1 = 0x01
        self.ACCEL_CONFIG = 0x14
        self.BANK_0 = 0b00000000
        self.BANK_2 = 0b00100000
        # GRYO - 0b00: 250dps, 0b01: 500dps, 0b10: 1000dps, 0b11: 2000dps
        self.GYRO_FS_MODE = 0b01 # DO NOT FORGET TO CHANGE THE SENSITIVITY SCALING FACTOR INSIDE scale_imu_readings()
        # ACCEL - 0b00: 2g, 0b01: 4g, 0b10: 8g, 0b11: 16g
        self.ACCEL_FS_MODE = 0b01 # DO NOT FORGET TO CHANGE THE SENSITIVITY SCALING FACTOR INSIDE scale_imu_readings()

        self.i2cbus = smbus.SMBus(self.JETSON_I2C_BUS)
        
        self.accel_gyro_offset = 0x2D
        self.accel_offsets = [0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32]
        self.gyro_offsets = [0x33, 0x34, 0x35, 0x36, 0x37, 0x38]

        self.wake_IMUs()
        self.set_gyro_full_scale()
        self.set_accel_full_scale()
    
    # def __del__(self): # try to close the bus when done just in case
    #     self.i2cbus.close() # doesn't actually seem to do anything

    def get_imu_address(self, port):
        return self.IMU_ADDRESS_soldered if port == self.IMU_mux_ports['IMU_PELVIS'] else self.IMU_ADDRESS

    def select_IMU(self, port):
        '''
        Selects IMU based on mux port id (read/write) \n
        Experimentation indicates that following a mux port switch\n
        write immediately with another i2cbus command often causes\n
        some sort of collision ("[Errno 121] Remote I/O error"),\n
        hence the time.sleep()
        '''
        self.i2cbus.write_byte(self.MULTIPLEXER_ADDRESS, port)
        

    def write_thru_mux(self, port, offset, data):
        '''
        Wrapper around smbus.i2cbus that also sets multiplexer port.\n
        Guarantees that port is always set, but at the time cost of an\n
        extra command to the mux each time.
        Once mux port is set, all IMUs are at the same address, so there's\n
        no need to set one.
        '''
        self.select_IMU(port)
        return self.i2cbus.write_byte_data(self.get_imu_address(port), offset, data)
    
    def read_thru_mux(self, port, offset):
        '''
        Wrapper around smbus.i2cbus that also sets multiplexer port.\n
        Guarantees that port is always set, but at the time cost of an\n
        extra command to the mux each time.
        Once mux port is set, all IMUs are at the same address, so there's\n
        no need to set one.
        '''
        self.select_IMU(port)
        return self.i2cbus.read_byte_data(self.get_imu_address(port), offset)

    def wake_IMU(self, port):
        '''
        Wakes up the IMU sensor to start reading values \n
        ONLY REQUIRED AT POWER UP (although re-sending occasionally\n
        without power cycling seems not to cause issues).\n
        Writes value 257 to PWR_MGMT at 0x06 and waits a little.
        '''
        self.write_thru_mux(port, 0x06, 0b00000101)
        time.sleep(1.5)

    def check_who_am_i(self, port):
        '''
        Returns the value stored in WHO_AM_I register \n
        Should return 234 (0xEA) if sensor is connected properly \n
        Use this to debug connection issues
        '''
        try:
            if self.read_thru_mux(port, self.WHO_AM_I) != self.YOU_ARE:
                print(f'IMU at mux port {port} connected but failed "whoami" check')
                return False
            return True
        except:
            print(f'IMU at mux port {port} not readable')
            return False

    def check_IMU_awake(self, port):
        '''
        Helper function for wake_IMUs
        '''
        awake = True

        if not self.check_who_am_i(port):
            awake = False

        elif np.sum(self.read_IMU(port)) == 0.0:
            print(f'IMU at port {port} read empty')
            awake = False

        return awake
    
    def wake_IMUs(self):
        '''
        Check if IMUs are on. If not, turn them on.
        '''
        # so can check all of them even if one's dead
        self.IMUs_are_on = None

        for port in self.IMU_mux_ports.values():
            print("Checking connection to IMU at mux port", port)
            
            if not self.check_IMU_awake(port):
                print('Sending wake signal')
                try:
                    self.wake_IMU(port)
                except Exception as imu_except:
                    print(f"Sending wake signal to IMU at {port} failed")
                    traceback.print_tb(imu_except.__traceback__)
                    self.IMUs_are_on = False
            
            if not self.check_IMU_awake(port):
                print('Failed to wake IMU')
                self.IMUs_are_on = False

        if self.IMUs_are_on is None:
            self.IMUs_are_on = True

    # note that this does *not* use write_thru_mux
    def get_imu_readings(self, port):
        '''
        Helper for getting gyro and acceleration data
        Reads data from the I2C bus at the given offsets
        '''
        self.select_IMU(port)
        
        data = self.i2cbus.read_i2c_block_data(
            self.get_imu_address(port), self.accel_gyro_offset, 12
        )

        # note to self: newbyteorder default is "'S' - swap dtype from
        # current to opposite endian"
        return np.frombuffer(bytes(data), dtype = np.int16).newbyteorder()
    
    def scale_imu_readings(self, data:np.ndarray):
        writeable_data = data.astype(float)

        writeable_data[0] *= -1 # Flipping sign for X-axis accelration (!!!!! DON'T CHANGE THIS !!!!!)
        writeable_data[4] *= -1 # Flipping sign for Y-axis gyro (!!!!! DON'T CHANGE THIS !!!!!)
        writeable_data[5] *= -1 # Flipping sign for Z-axis gyro (!!!!! DON'T CHANGE THIS !!!!!)
        
        # ACCEL_FS=0 : 16384.0 / ACCEL_FS=1 8192.0 / ACCEL_FS=2 : 4096.0 / ACCEL_FS=3 2048.0
        writeable_data[:3] = 9.80665 * writeable_data[:3] / 8192.0 # acceleration
        # GYRO_FS_SEL=0 : 131.0 / GYRO_FS_SEL=1 : 65.5 / GYRO_FS_SEL=2 : 32.8 / GYRO_FS_SEL=3 : 16.4
        writeable_data[3:] = writeable_data[3:] / 65.5 # gyro

        return writeable_data
    
    # note that this does *not* use write_thru_mux
    def read_IMU(self, port):
        '''
        Get linear and angular accelerations from a given IMU
        return format: array - [linx, liny, linz, angx, angy, angz] 
        '''
        return self.scale_imu_readings(self.get_imu_readings(port))

    def read_IMUs(self):
        '''
        Get linear and angular accelerations from all IMUs
        '''
        for name, port in self.IMU_mux_ports.items():
            self.imu_readings[name] = self.read_IMU(port)
        return self.imu_readings

    def set_gyro_full_scale(self):
        for port in self.IMU_mux_ports.values():
            self.write_thru_mux(port, self.BANK_SEL, self.BANK_2)
            current_value = self.read_thru_mux(port, self.GYRO_CONFIG_1)
            new_value = (current_value & 0b11111001) | (self.GYRO_FS_MODE << 1)
            self.write_thru_mux(port, self.GYRO_CONFIG_1, new_value)
            time.sleep(0.1)
            self.write_thru_mux(port, self.BANK_SEL, self.BANK_0)

    def set_accel_full_scale(self):
        for port in self.IMU_mux_ports.values():
            self.write_thru_mux(port, self.BANK_SEL, self.BANK_2)
            current_value = self.read_thru_mux(port, self.ACCEL_CONFIG)
            new_value = (current_value & 0b11111001) | (self.ACCEL_FS_MODE << 1)
            self.write_thru_mux(port, self.ACCEL_CONFIG, new_value)
            time.sleep(0.1)
            self.write_thru_mux(port, self.BANK_SEL, self.BANK_0)

def main():
    imus = ICM20948_I2C_IMUs()

    # check IMU readings
    while imus.IMUs_are_on:
        for key, val in imus.read_IMUs().items():
            print(key[-5:], np.round(val, 2), end="  ")
        print()
        time.sleep(1)

    # check averages
    start = time.time()
    iters = 0

    # # check average IMU readings
    # sum_reads = imus.read_IMUs()
    # # check IMU readings
    # while imus.IMUs_are_on:
    #     iters += 1
    #     for key, val in sum_reads.items():
    #         print(key, np.round(val/iters, 2))
    #     print()
    #     # time.sleep(1)
    #     for key, val in imus.read_IMUs().items():
    #         sum_reads[key] += val

    # while imus.IMUs_are_on:
    #     for key, val in imus.read_IMUs().items():
    #         print(key[-5:], np.round(val, 2), end="    ")
    #     print()
    #     iters += 1
    #     print((time.time() - start)/iters)

if __name__ == '__main__':
    main()