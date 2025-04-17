/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "platform/mbed_thread.h"


/*   C 	     C#	     D	     Eb	    E	      F	      F#	  G	      G#	   A	Bb	     B
0	16.35	17.32	18.35	19.45	20.60	21.83	23.12	24.50	25.96	27.50	29.14	30.87
1	32.70	34.65	36.71	38.89	41.20	43.65	46.25	49.00	51.91	55.00	58.27	61.74
2	65.41	69.30	73.42	77.78	82.41	87.31	92.50	98.00	103.8	110.0	116.5	123.5
3	130.8	138.6	146.8	155.6	164.8	174.6	185.0	196.0	207.7	220.0	233.1	246.9
4	261.6	277.2	293.7	311.1	329.6	349.2	370.0	392.0	415.3	440.0	466.2	493.9
5	523.3	554.4	587.3	622.3	659.3	698.5	740.0	784.0	830.6	880.0	932.3	987.8
6	1047	1109	1175	1245	1319	1397	1480	1568	1661	1760	1865	1976
7	2093	2217	2349	2489	2637	2794	2960	3136	3322	3520	3729	3951
8	4186	4435	4699	4978	5274	5588	5920	6272	6645	7040	7459	7902*/


#include "mbed.h"

PwmOut volume(D6);
Ticker note_player;


const float REST = 0.0;

//initialize all note values for 8 octaves so you can add more songs
const float C0  = 16.35;
const float Cs0 = 17.32; 
const float D_0  = 18.35;
const float Ds0 = 19.45; 
const float E0  = 20.60;
const float F0  = 21.83;
const float Fs0 = 23.12;
const float G0  = 24.50;
const float Gs0 = 25.96; 
const float A_0  = 27.50;
const float As0 = 29.14; 
const float B0  = 30.87;

//first octave
const float C1  = 32.70;
const float Cs1 = 34.65;
const float D_1  = 36.71;
const float Ds1 = 38.89;
const float E1  = 41.20;
const float F1  = 43.65;
const float Fs1 = 46.25;
const float G1  = 49.00;
const float Gs1 = 51.91;
const float A_1  = 55.00;
const float As1 = 58.27;
const float B1  = 61.74;

//second octave
const float C2  = 65.41;
const float Cs2 = 69.30;
const float D_2  = 73.42;
const float Ds2 = 77.78;
const float E2  = 82.41;
const float F2  = 87.31;
const float Fs2 = 92.50;
const float G2  = 98.00;
const float Gs2 = 103.83;
const float A_2  = 110.00;
const float As2 = 116.54;
const float B2  = 123.47;

//third octave
const float C3  = 130.81;
const float Cs3 = 138.59;
const float D_3  = 146.83;
const float Ds3 = 155.56;
const float E3  = 164.81;
const float F3  = 174.61;
const float Fs3 = 185.00;
const float G3  = 196.00;
const float Gs3 = 207.65;
const float A_3  = 220.00;
const float As3 = 233.08;
const float B3  = 246.94;

//fourth octave
const float C4  = 261.63;
const float Cs4 = 277.18;
const float D_4  = 293.66;
const float Ds4 = 311.13;
const float E4  = 329.63;
const float F4  = 349.23;
const float Fs4 = 369.99;
const float G4  = 392.00;
const float Gs4 = 415.30;
const float A_4  = 440.00;
const float As4 = 466.16;
const float B4  = 493.88;

//fifth octave
const float C5  = 523.25;
const float Cs5 = 554.37;
const float D_5  = 587.33;
const float Ds5 = 622.25;
const float E5  = 659.25;
const float F5  = 698.46;
const float Fs5 = 739.99;
const float G5  = 783.99;
const float Gs5 = 830.61;
const float A_5  = 880.00;
const float As5 = 932.33;
const float B5  = 987.77;

//sixth octave
const float C6  = 1046.50;
const float Cs6 = 1108.73;
const float D_6  = 1174.66;
const float Ds6 = 1244.51;
const float E6  = 1318.51;
const float F6  = 1396.91;
const float Fs6 = 1480.00;
const float G6  = 1567.98;
const float Gs6 = 1661.22;
const float A6  = 1760.00;
const float As6 = 1864.66;
const float B6  = 1975.53;

// seventh octave
const float C7  = 2093.00;
const float Cs7 = 2217.46;
const float D_7  = 2349.32;
const float Ds7 = 2489.02;
const float E7  = 2637.02;
const float F7  = 2793.83;
const float Fs7 = 2959.96;
const float G7  = 3135.96;
const float Gs7 = 3322.44;
const float A7  = 3520.00;
const float As7 = 3729.31;
const float B7  = 3951.07;

//eight octave
const float C8  = 4186.01;
const float Cs8 = 4434.92;
const float D_8  = 4698.63;
const float Ds8 = 4978.03;
const float E8  = 5274.04;
const float F8  = 5587.65;
const float Fs8 = 5919.91;
const float G8  = 6271.93;
const float Gs8 = 6644.88;
const float A8  = 7040.00;
const float As8 = 7458.62;
const float B8  = 7902.13;





const 

//array that lists the notes to play in order
float melody[] = {
   
    REST, REST, REST, Ds4,
    E4, REST, Fs4, G4, REST, Ds4,
    E4, Fs4, G4, C5, B4, E4,
    G4, B4, As4, A_4, G4, E4,
    D_4, E4,
    
   
    REST, REST, REST, Ds4,
    E4, REST, Fs4, G4, REST, Ds4,
    E4, Fs4, G4, C5, B4, E4,
    G4, B4, As4, A_4, G4, E4,
    D_4, E4,
    
   
    REST, REST, E5, D_5, 
    B4, A_4, G4, E4,
    As4, A_4, As4, A_4, 
    As4, A_4, As4, G4, 
    E4, D_4, E4, E4, E4
};

//array that holds the beat duration for each note
float note_beats[] = {
    //main panther theme
    2,2,2,4,
    2,2,2,8,4, 4,
    2, 2, 8, 4, 2, 4,
    2, 4, 2, 2, 2, 2,
    2, 8,
    
    //repeat main
    2, 2, 2, 4,
    2, 2, 2, 8, 4, 4,
    2,2, 8,4, 2, 4,
    2, 4, 2, 2, 2, 2,
    2, 8,
    
    //final theme
    4,2, 4, 2, 
    2, 2,2, 4,
    2, 2, 2, 2, 
    2,2,4, 2, 
    2, 2, 8, 4, 8
};

//index for note and melody arrays
int k = 0;                   
int totalNotes;             
const float tempo = 1.5;    

//ISR TICKER
void ISR_play_note() {
    //conditional statement will transition through the melody
    if (k < totalNotes) 
    {
        //if the note in melody is equal to REST there should be no sound
        if (melody[k] == REST) 
        {
            volume = 0.0;
        } 
        else 
        {
             if (melody[k] > 0) //if the note is anything but rest set the period
            {
                volume.period(1.0 / melody[k]);
            }
            //use 50% duty cycle for a clearer tone
            volume = 0.5;
        }
        float beat_time = (note_beats[k] * tempo) / 32.0; 
        
         //restart the ISR ticker for the next note in the melody
        note_player.detach();
        note_player.attach(&ISR_play_note, beat_time);
        
        k++;//next note
    } 
    
    else //restart
    {
        volume = 0.0;  
        k = 0;       
        note_player.detach();
        note_player.attach(&ISR_play_note, 1.0);  //sets a small pause before restarting
    }
}

int main() {
    totalNotes = sizeof(melody) / sizeof(melody[0]);
    volume = 0.0;
    
    note_player.attach(&ISR_play_note, 0.1);  
}
