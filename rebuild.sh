docker-compose down
#docker rmi ghx-baremetal-backend ghx-frontend
docker build -f Dockerfile.ghx-dashboard -t ghx-frontend .
docker build -f Dockerfile.ghx-backend -t ghx-baremetal-backend .
docker-compose up -d
docker-compose ps