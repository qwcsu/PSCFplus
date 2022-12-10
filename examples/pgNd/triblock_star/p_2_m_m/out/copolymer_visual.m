function copolymer_visual(filename)

C = textread(filename, '%s','delimiter', '\n');
dim = str2num(C{3});                    % Read dimension
cell = str2num(C{9});                   % Read cell parameter
grid = str2num(C{15});                  % Read grid

if(length(cell)==1)
   new_cell = ones(1,3)*cell;           % Cubic crystals 
elseif(length(cell)==2)
    new_cell(1:2) = cell(1);            % Tetragonal crystals
    new_cell(3)   = cell(2);
else
    new_cell = cell;                    % Orthorhombic crystals
end
 
if(length(grid)==1)
   grid(2) = grid(1);                   % 3D grid for 1D crystals
   grid(3) = grid(1);
elseif(length(grid)==2)
    grid(3) = grid(1);                  % 3D grid for 2D crystals
end

clear cell;cell = new_cell;
for i =16:length(C)    
    A(i-15,:) = str2num(C{i});
end
v1 = A(:,1); v2 = A(:,2);

% Formulating the volume fraction in 3D  arrays for 3D visualization.
clear x y z a b c b2 A;
counter = 0;
for iz=1:grid(3),
    for iy=1:grid(2),
        for ix=1:grid(1),
            counter = counter + 1;
            x(ix,iy,iz) = cell(1) * (ix-1)/grid(1);
            y(ix,iy,iz) = cell(2) * (iy-1)/grid(2);
            z(ix,iy,iz) = cell(3) * (iz-1)/grid(3);
            a(ix,iy,iz) = v1(counter);
            b(ix,iy,iz) = v2(counter);
        end
        if(dim==1)
           counter = 0;
        end
    end
    if(dim==2)
        counter=0;
    end
end
    
box on; hold on;

% Volume fraction of A monomer type.
data = smooth3(a,'box',5);
isovalue =0.3;
p1 = patch(isosurface(x,y,z,data,isovalue), ...
     'FaceColor','blue','EdgeColor','none');
p2 = patch(isocaps(x,y,z,data,isovalue), ...
     'FaceColor','interp','EdgeColor','none');
isonormals(data,p1);
axis equal;
view(3); axis vis3d tight
lightangle(45,30);
set(gcf,'Renderer','zbuffer'); lighting phong
set(p2,'AmbientStrength',.6); 
set(p1,'AmbientStrength',.5); 


% Volume fraction of B monomer type.
% data = smooth3(b,'box',5);
% isovalue =0.5;
% p1 = patch(isosurface(x,y,z,data,isovalue), ...
%      'FaceColor','red','EdgeColor','none');
% p2 = patch(isocaps(x,y,z,data,isovalue), ...
%      'FaceColor','interp','EdgeColor','none');
% isonormals(data,p1);
% axis equal;
% view(3); axis vis3d tight
% lightangle(45,30);
% set(gcf,'Renderer','zbuffer'); lighting phong
% set(p2,'AmbientStrength',.6); 
% set(p1,'AmbientStrength',.5); 

end
