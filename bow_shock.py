import numpy as np 
import scipy.misc as smp
import numba as nb
from scipy import ndimage, misc	
import random 
from PIL import Image, ImageDraw, ImageFilter
import glob
from scipy.ndimage import gaussian_filter
import os
from skimage.util import random_noise

def calc_flux(r0, scale, qincl):
	mx, my =201, 201
	dim = 0 # 0 is isotropic
	#qincl = 30.*np.pi/180. #inclination of the bow shock system
	nphi, nth = 300, 300
	ph1, ph2 = 0, 7.*np.pi/8.
	ixs, iys = 150, 100
	dphi = (ph2-ph1)/nphi
	dth = 2.*np.pi/nth
	isot = True
	phi_t = ph1 + dphi*np.arange(1, nphi) #create grid of phi
	#classic sperical wind
	#aligned cos wind Wilkin (2000)
	c1=0.
	if (isot==True):
		r_t=(r0/np.sin(phi_t))*((3.*(1-phi_t/np.tan(phi_t)))**0.5)
	else:
		c2 = 1
		r_t=(r0/np.sin(phi_t))*( 3.*(1-phi_t)*(1.-c2/12.) +2.*c1*(1.-np.cos(phi_t)) + 3.*c2*np.sin(phi_t)**2/4. )**0.5

	data = np.zeros((nphi, nth))

	@nb.jit(fastmath=True)
	def calc_flux(data, phi_t, r_t, qincl, dth):
		for i in range(data.shape[0]-1):
			phi = phi_t[i]
			r = r_t[i]
			for j in range (data.shape[1]-1):  #! theta loop
				th=-np.pi+float(j)*dth 
				czeta=np.cos(phi)*np.cos(qincl)+np.sin(phi)*np.sin(qincl)*np.cos(th)
				schi=np.sin(phi)*np.sin(th)/(1.-czeta*czeta)**0.5

				sw=np.sin(th)*np.sin(qincl)/np.sin(phi)
				szeta=np.sin(th)*np.sin(phi)/schi
				cchi= (np.cos(phi)-czeta*np.cos(qincl))/(szeta*np.sin(qincl))

				chi=np.arcsin(schi)
				if (cchi<0):
					chi= np.pi-chi
				zeta=np.arccos(czeta)

				xx=r*np.sin(zeta)*np.cos(chi)
				yy=r*np.sin(zeta)*np.sin(chi)

				ix=(xx*scale)+ixs #scaling
				if(ix > mx or ix<1):
					continue
				iy=(yy*scale)+iys #scaling
				if(iy > my or iy<1):
					continue
				amp=r*np.sin(phi)
				amp=amp*(1.+0.1*np.exp(-(phi-ph2)**2./2./0.003))
				data[int(ix),int(iy)]= data[int(ix),int(iy)] + amp
		return data
	data = calc_flux(data, phi_t, r_t, qincl, dth)	
	return data

def translate(data, n):
	'''
	translate data by n pixels
	'''	
	data = np.roll(data, n, axis=0)
	data = np.roll(data, n, axis=1)
	return data

def rotate(data, angle):
	data = ndimage.rotate(data, angle, reshape=False)
	return data

def save_img(data, i, j):
	data = gaussian_filter(data, sigma = 3) #blur the image by 5 pixels gaussian width
	img = smp.toimage(data)       # Create a PIL image
	smp.imsave("Bow_shocks/bs"+str(i)+str(j)+'.png', img)

def salt_pepper(image, amount):
	'''
	Add salt and pepper noise to the image
	'''
	row,col = image.shape
	s_vs_p = 0.5
	out = image
	# Salt mode
	num_salt = np.ceil(amount * image.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape]
	out[coords] = 1

	# Pepper mode
	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))
			  for i in image.shape]
	out[coords] = 0
	return out

def crop_img():
	img_m = glob.glob('Real_images/*.png')#os.listdir("Real_images")
	for k in range(len(img_m)):
		img = smp.imread(img_m[k])
		img = np.asarray(img)
		i = 0
		for r in range(0,img.shape[0],150):
			j = 0
			for c in range(0,img.shape[1],300):
				data = img[r:r+300, c:c+300][:,:,0]
				data1 = smp.toimage(data)
				if (data.shape==(300, 300)):
					smp.imsave("Background/bkg"+str(k)+str(i)+str(j)+'.png', data1)
				j += 1
			i += 1

def paste_img():
	im1 = glob.glob("Training_data/Bkg/*.png")
	im2 = glob.glob("Training_data/Bow_shocks/*.png")
	#im1 = Image.open('Training_data/Bbkg000.png')
	#im2 = Image.open('bs11.png')
	#img = Image.blend(im1, im2, 0.15)
	#img.save('positive01i.png')
	#sys.exit()
	for i in range(len(im1)):
		image_1 = Image.open(im1[i])
		alpha = random.uniform(0.3, 0.6)
		image_2 = Image.open(im2[i])
		img = Image.blend(image_1, image_2, alpha)
		img.save('Training_data/Positive/'+str(i)+'.png')


def main1():
	for i in range(26, 28):
		for j in range(26):
			r0 = random.uniform(0.5, 3.5)
			n = random.randint(20, 300)
			angle = random.uniform(0.0, 1.0)*360.
			qincl = random.uniform(0.17, 1.56)
			data = calc_flux(r0, 10, qincl)
			data = np.abs(rotate(data, angle))
			noise_mask = np.random.poisson(data)	
			data = data + noise_mask
			amount = random.uniform(0.5, 1.0)
			data = salt_pepper(data, amount)			
			save_img(data, i, j)

def main():
	#crop_img()
	paste_img()

if __name__ == '__main__':
	main()