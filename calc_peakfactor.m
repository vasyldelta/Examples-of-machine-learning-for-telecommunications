function p=calc_peakfactor(x,pos)
p=x(pos)/mean(x([pos-10:pos-4 pos+4:pos+10]));
